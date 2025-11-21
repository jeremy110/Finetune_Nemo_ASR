import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import os

def merge_tokenizer_models_with_vocab(
    model_path1: str, 
    model_path2: str, 
    output_model_path: str, 
    vocab_output_path: str
):
    """
    Merges the vocabularies of two SentencePiece models.

    Model 1 is treated as the primary model (e.g., English), retaining its UNKNOWN token.
    Model 2 is treated as the secondary/auxiliary model (e.g., Chinese/Hi), 
    with its first piece (usually UNK) skipped and scores adjusted by -1024.

    Args:
        model_path1 (str): Path to the first (primary) SentencePiece model file.
        model_path2 (str): Path to the second (auxiliary) SentencePiece model file.
        output_model_path (str): Path to save the new merged .model file.
        vocab_output_path (str): Path to save the new merged .vocab file.
    """
    print(f"--- Starting Model Merge ---")
    print(f"Primary Model (Model 1): {model_path1}")
    print(f"Auxiliary Model (Model 2): {model_path2}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(vocab_output_path) or '.', exist_ok=True)

    # Load and parse the two models
    sp1 = spm.SentencePieceProcessor(model_file=model_path1)
    sp2 = spm.SentencePieceProcessor(model_file=model_path2)

    model_proto1 = sp_pb2_model.ModelProto()
    model_proto2 = sp_pb2_model.ModelProto()
    model_proto1.ParseFromString(sp1.serialized_model_proto())
    model_proto2.ParseFromString(sp2.serialized_model_proto())

    # Initialize new structure for the merged model
    new_model_proto = sp_pb2_model.ModelProto()
    vocab_set = set() # For quick look-up and deduplication
    vocab_list = []   # To maintain order and save the final .vocab file

    def add_pieces_to_model_proto(source_proto, target_proto, vocab_set, vocab_list, is_auxiliary=False):
        """Adds pieces from the source model to the target proto, handling deduplication and score adjustments."""

        for idx, piece in enumerate(source_proto.pieces):
            p1 = piece.piece
            
            # --- Model 2 (Auxiliary) Specific Logic (zh=True in original script) ---
            if is_auxiliary:
                # 1. Skip the first piece (usually <unk> or padding) since Model 1 handles the special tokens.
                if idx == 0:
                    continue
                # 2. Adjust score by subtracting 1024 (as per original script's logic for the auxiliary model)
                p_score = piece.score - 1024
            # --- Model 1 (Primary) Specific Logic (zh=False in original script) ---
            else:
                # 1. (Original script's commented-out logic: p1 = piece.piece.upper() for non-special tokens)
                #    Keeping the original piece.piece value here as it was in the execution path.
                if idx != 0:
                    p1 = piece.piece 
                # 2. Use the original score for the primary model
                p_score = piece.score
            
            # Deduplication check
            if p1 not in vocab_set:
                vocab_set.add(p1)
                vocab_list.append((p1, p_score))
                
                # Add the piece to the new model proto
                new_piece = target_proto.pieces.add()
                new_piece.piece = p1
                new_piece.score = p_score
                
                # Set type for the first piece of Model 1 (usually UNKNOWN)
                if idx == 0 and not is_auxiliary:
                    new_piece.type = sp_pb2_model.ModelProto.SentencePiece.Type.UNKNOWN
                # For other tokens, inherit the type from the source piece
                else:
                    new_piece.type = piece.type

    # 4. Add pieces from the primary model (Model 1)
    print(f"  [Model 1] Adding pieces: {len(model_proto1.pieces)}")
    add_pieces_to_model_proto(model_proto1, new_model_proto, vocab_set, vocab_list, is_auxiliary=False)
    
    # 5. Add pieces from the auxiliary model (Model 2)
    print(f"  [Model 2] Adding pieces: {len(model_proto2.pieces)}")
    add_pieces_to_model_proto(model_proto2, new_model_proto, vocab_set, vocab_list, is_auxiliary=True)

    print(f"  Merge successful. Total vocabulary size: {len(new_model_proto.pieces)}")

    # 6. Set Trainer and Normalizer specs (Merging with Model 2's settings, as per original script)
    # Note: Using MergeFrom() might override some settings but keeps the core structure.
    print("  Merging Trainer and Normalizer specs from Model 2...")
    # print(model_proto1.trainer_spec, model_proto2.trainer_spec) # Original print statement
    new_model_proto.trainer_spec.MergeFrom(model_proto2.trainer_spec)
    new_model_proto.normalizer_spec.MergeFrom(model_proto2.normalizer_spec)
    
    # Crucial: Update the final vocab size in the trainer spec
    new_model_proto.trainer_spec.vocab_size = len(new_model_proto.pieces)


    # 7. Save the new model file
    with open(output_model_path, 'wb') as f:
        f.write(new_model_proto.SerializeToString())
    print(f"✅ New merged tokenizer saved to {output_model_path}")

    # 8. Save the vocabulary file (piece <TAB> score)
    with open(vocab_output_path, 'w', encoding='utf-8') as f:
        for piece, score in vocab_list:
            # SentencePiece vocab format is token + TAB + score
            f.write(f"{piece}\t{score}\n") 
    print(f"✅ Vocabulary saved to {vocab_output_path}")
    print(f"--- Merge Complete ---")


# --- Example Usage (requires actual model files at these paths) ---

# Define the paths to your input models
en_path = 'a4c254472f6a4c388105ae0109636cdc_tokenizer.model' # original nemo en tokenizer
zh_path = './hi/tokenizer.model' # using https://github.com/AI4Bharat/IndicVoices/tree/master/artifacts/tokenizers/hi_256/tokenizer_spe_bpe_v256

# Define the output directory and file names
output_dir = r".\en1024_hi256"
output_model = os.path.join(output_dir, "tokenizer.model")
output_vocab = os.path.join(output_dir, "tokenizer.vocab")

# Execute the merge function
merge_tokenizer_models_with_vocab(
    en_path, 
    zh_path, 
    output_model, 
    output_vocab
)

# Load and verify the merged model
print("\n--- Verifying Merged Model ---")
sp = spm.SentencePieceProcessor()
sp.load(output_model)
print(f"Merged Vocabulary Size: {sp.get_piece_size()}")
print(f"Test encoding 'Hello world! क्या आप ठीक हैं': {sp.encode_as_ids('Hello world! क्या आप ठीक हैं')}")