import os
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load

_has_cuda = torch.cuda.is_available()

if _has_cuda:
    CHUNK_LEN = 16
    HEAD_SIZE = 64
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name="wind_backstepping", sources=[f'{this_file_path}/wkv7_cuda.cu', f'{this_file_path}/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, z, b):
            B, T, H, C = w.shape 
            assert T % CHUNK_LEN == 0
            assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
            assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
            y = torch.empty_like(v)
            s = torch.empty(B, H, T//CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
            sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
            torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
            ctx.save_for_backward(w, q, k, v, z, b, s, sa)
            return y
        
        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype == torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w, q, k, v, z, b, s, sa = ctx.saved_tensors
            dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
            torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
            return dw, dq, dk, dv, dz, db

    def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
        B, T, HC = q.shape
        q, w, k, v, a, b = [i.view(B, T, HC // 64, 64).bfloat16().contiguous() for i in [q, w, k, v, a, b]]

        return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)
else:
    # DTYPE = torch.bfloat16
    def RWKV7_OP_mask(r, w, k, v, a, b, HEAD_SIZE = 64):
        B, T, C = r.size()
        H = C // HEAD_SIZE
        N = HEAD_SIZE
        
        # 轉換為 float 進行計算以提高數值穩定性
        r, w, k, v, a, b = [i.view(B, T, H, N).float() for i in [r, w, k, v, a, b]]
        
        # 計算權重衰減
        w = torch.exp(-torch.exp(w))
        
        # 初始化輸出和狀態
        out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
        state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)
            
        # 逐時間步計算
        for t in range(T):
            kk = k[:, t, :].view(B, H, 1, N)     # (B, H, 1, N)
            rr = r[:, t, :].view(B, H, N, 1)     # (B, H, N, 1)
            vv = v[:, t, :].view(B, H, N, 1)     # (B, H, N, 1)
            aa = a[:, t, :].view(B, H, N, 1)     # (B, H, N, 1)
            bb = b[:, t, :].view(B, H, 1, N)     # (B, H, 1, N)
            
            # 狀態更新：state = state * w + state @ a @ b + v @ k
            w_t = w[:, t, :, None, :]  # (B, H, 1, N) -> (B, H, N, N) for broadcasting
            state = state * w_t + state @ aa @ bb + vv @ kk
            
            # 計算輸出
            out[:, t, :] = (state @ rr).view(B, H, N)

        return out.view(B, T, C)#.to(dtype=DTYPE)




class BiRWKV7TimeMix(nn.Module):
    '''
        Replace the NeMo RelPositionMultiHeadAttention with the BiRWKV7TimeMix block.
        Then remove the original positional encoding.
        For the parameters, reinitialize them instead of directly loading the original NeMo weights.
        Use ZeroPad2d for the time_shift as before, rather than the convolution used in AudioRWKV.
        I’m not sure whether the masking part is handled correctly.
    '''

    def __init__(
        self, 
        n_head,
        n_feat,
        layer_id,
        n_layer,
    ):
        super().__init__()
        hidden_size = n_feat
        self.n_head = n_head
        self.layer_id = layer_id
        self.head_size = hidden_size // n_head
        
        H = self.n_head
        N = self.head_size
        C = n_feat
        
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x
                
            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            D_DECAY_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)

            D_MV_LORA = max(32, int(round(  (1.7*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round(  (5*(C**0.5))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.gate = nn.Conv1d(
                in_channels = C,
                out_channels = H,
                kernel_size = 1,
                groups = H,
                bias = False
            )

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            # self.time_shift = nn.Conv1d(
            #     in_channels = C,
            #     out_channels = C,
            #     kernel_size = 3,
            #     padding = 1,
            #     groups = C,
            #     bias = False
            # )
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5, dtype=torch.bfloat16)
            # self.ln_x.float()
            
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)


            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.time_shift.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

        del ddd, www, zigzag, linear

    def forward(
            self, 
            x: torch.Tensor, 
            v_first: torch.Tensor, 
            mask = None
    ):
        '''
            x: hidden_states
                (B, T, C)
            mask: for padding mask -> output like: [False, False, False, True]
                (B, T)
        
        '''
        # 如果使用 cuda 運算, 長度要是 16 的倍數
        input_seq_len = x.size(1)
        
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = (~mask).float()
            else:
                mask = 1.0 - mask.float()

        if input_seq_len % 16 != 0:
            pad_len = 16 - input_seq_len % 16
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                # padding 部分的 mask 設為 0
                mask = F.pad(mask, (0, pad_len), value=0)
            if v_first is not None:
                v_first = F.pad(v_first, (0, 0, 0, pad_len))

        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x
        # xx = self.time_shift(x.permute(0,2,1)).permute(0,2,1) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        # ===== test apply mask to k, v, w =====
        if mask is not None:
            # [B, T] -> [B, T, 1]
            mask_expanded = mask.float().unsqueeze(-1)
            
            # 遮蔽 padding 部分的 k 和 v
            k = k * mask_expanded  # [B, T, C] * [B, T, 1]
            v = v * mask_expanded
            
            # 對於 w (decay), padding 部分設為極大負值，使其完全衰減
            # 這樣 padding 不會影響 state 累積
            w = w * mask_expanded + (1.0 - mask_expanded) * (-1000.0)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2)

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        # RWKV7 運算
        if _has_cuda:
            # 雙向處理
            r_cat = torch.cat([r, torch.flip(r, dims=[1])], dim=0)
            w_cat = torch.cat([w, torch.flip(w, dims=[1])], dim=0)
            k_cat = torch.cat([k, torch.flip(k, dims=[1])], dim=0)
            v_cat = torch.cat([v, torch.flip(v, dims=[1])], dim=0)
            kk_cat = torch.cat([-kk, torch.flip(-kk, dims=[1])], dim=0)
            kka_cat = torch.cat([(kk*a), torch.flip(kk*a, dims=[1])], dim=0)
            x_out = RUN_CUDA_RWKV7g(r_cat, w_cat, k_cat, v_cat, kk_cat, kka_cat)
            
            x_out = x_out.view(2 * B, T, H, -1)
            x_f, x_b = torch.chunk(x_out, chunks=2, dim=0)
            
            gate = torch.sigmoid(self.gate(xx.transpose(1, 2))).transpose(1, 2).unsqueeze(3)
            x = gate * x_f + (1.0 - gate) * torch.flip(x_b, dims=[1])
            
        else:
            # 目前只測試 cuda kernel 正常運行
            # x = RWKV7_OP_mask(r, w, k, v, -kk, kk * a, mask)
            pass
            
        # 使用 GroupNorm 對輸出進行歸一化
        x = self.ln_x(x.contiguous().view(B * T, C)).view(B, T, C)

        if mask is not None:
            x = x * mask_expanded
        
        x = x + ((r.view(B, T, H, -1) * 
                  k.view(B, T, H, -1) * 
                  self.r_k).sum(dim = -1, keepdim = True) 
                  * v.view(B, T, H, -1)).view(B, T, -1)

        # 最終輸出經過門控 g 控制, 並通過線性層輸出
        x = self.output(x * g)

        if input_seq_len != T:
            x = x[:, :input_seq_len]
            v_first = v_first[:, :input_seq_len]

        return x, v_first