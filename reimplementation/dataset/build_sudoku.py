import os
import csv
import json
import argparse
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from .common import PuzzleDatasetMetadata

def parse_args():
    parser = argparse.ArgumentParser(description="Build Sudoku Dataset")
    parser.add_argument("--source_repo", type=str, default="sapientinc/sudoku-extreme")
    parser.add_argument("--output_dir", type=str, default="data/sudoku-extreme-reimpl")
    parser.add_argument("--subsample_size", type=int, default=None)
    parser.add_argument("--min_difficulty", type=int, default=None)
    parser.add_argument("--num_aug", type=int, default=0)
    return parser.parse_args()

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    """
    Applies Sudoku-preserving transformations:
    1. Permute digits (1-9)
    2. Transpose (maybe)
    3. Permute rows within bands (3 groups of 3 rows)
    4. Permute bands
    5. Permute columns within stacks
    6. Permute stacks
    """
    # 1. Digit permutation (0 stays 0)
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # 2. Transpose
    transpose_flag = np.random.rand() < 0.5

    # 3 & 4. Row permutations
    # Permute bands (0,1,2), then permute rows within each band
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # 5 & 6. Column permutations
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Construct the mapping for the flattened array or indices
    # We can just index the array directly
    
    def apply_transformation(x: np.ndarray) -> np.ndarray:
        if transpose_flag:
            x = x.T
        
        # Apply row and col permutations
        x = x[row_perm, :]
        x = x[:, col_perm]
        
        # Apply digit mapping
        return digit_map[x]

    return apply_transformation(board), apply_transformation(solution)

def process_subset(set_name: str, args):
    print(f"Processing {set_name}...")
    
    # Download and read CSV
    csv_path = hf_hub_download(repo_id=args.source_repo, filename=f"{set_name}.csv", repo_type="dataset")
    
    inputs = []
    labels = []
    
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for source, q, a, rating in reader:
            if args.min_difficulty is None or int(rating) >= args.min_difficulty:
                # Convert string '003...' to numpy array
                # q is the puzzle (zeros are blanks), a is the solution
                q_arr = np.array([int(c) for c in q.replace('.', '0')], dtype=np.uint8).reshape(9, 9)
                a_arr = np.array([int(c) for c in a], dtype=np.uint8).reshape(9, 9)
                inputs.append(q_arr)
                labels.append(a_arr)
    
    # Subsample if needed (only for training usually)
    if set_name == "train" and args.subsample_size is not None and args.subsample_size < len(inputs):
        indices = np.random.choice(len(inputs), size=args.subsample_size, replace=False)
        inputs = [inputs[i] for i in indices]
        labels = [labels[i] for i in indices]
        
    # Augmentation
    num_augments = args.num_aug if set_name == "train" else 0
    
    final_inputs = []
    final_labels = []
    puzzle_indices = [] # Map example to original puzzle index
    group_indices = [0] # Start of each group of augmentations
    puzzle_identifiers = [] # Just for compatibility, seemingly unused in simple case
    
    puzzle_id = 0
    example_id = 0
    
    for orig_inp, orig_out in tqdm(zip(inputs, labels), total=len(inputs)):
        # Original
        final_inputs.append(orig_inp)
        final_labels.append(orig_out)
        puzzle_indices.append(example_id)
        puzzle_identifiers.append(0)
        example_id += 1
        
        # Augmentations
        for _ in range(num_augments):
            aug_inp, aug_out = shuffle_sudoku(orig_inp, orig_out)
            final_inputs.append(aug_inp)
            final_labels.append(aug_out)
            puzzle_indices.append(example_id)
            puzzle_identifiers.append(0)
            example_id += 1
            
        puzzle_id += 1
        group_indices.append(puzzle_id) # This looks wrong in my logic vs original? 
        # Original: results["group_indices"].append(puzzle_id) inside the loop? 
        # Wait, let's check original logic carefully.
        
    # Re-checking original logic for group_indices:
    # results["group_indices"].append(0) (init)
    # Loop puzzles:
    #   Loop augments:
    #     add input/label
    #     add puzzle_indices -> example_id
    #     example_id++
    #   puzzle_id++
    #   results["group_indices"].append(puzzle_id)
    #
    # Wait, group_indices seems to track the *index in the puzzle list* (puzzle_id), not the index in the *example list*.
    # Actually, in many datasets group_indices usually points to the start/end in the flat list.
    # Let's look at `common.py` or usage. But if I follow the original code strictly:
    # `results["group_indices"]` stores `puzzle_id`. 
    # Ah, `total_groups = len(results["group_indices"]) - 1`
    # It seems `group_indices` is just `[0, 1, 2, ..., N]`. This seems redundant if it's just range(N+1).
    # UNLESS it is used to group examples.
    # In the original code: `results["group_indices"].append(puzzle_id)` is called *after* the augment loop.
    # And `puzzle_id` increments by 1 per original puzzle.
    # So `group_indices` will be `[0, 1, 2, 3, ...]`. 
    # This implies that `group_indices` might not be "start indices in the flat array" but just "puzzle IDs".
    # BUT usually `group_indices` in these datasets (like in FAIR's code) implies boundaries.
    # Let's double check if I missed something in original code.
    # Original: `puzzle_id` starts at 0. Loop: ... puzzle_id += 1; results["group_indices"].append(puzzle_id).
    # So it is indeed just 0, 1, 2...
    # However, `puzzle_indices` maps each *example* to `example_id`.
    # `puzzle_identifiers` maps each *example* to `0`.
    
    # Wait, `puzzle_indices` in original:
    # `results["puzzle_indices"].append(0)` (init)
    # Loop:
    #   results["puzzle_indices"].append(example_id) (inside aug loop)
    # This array grows with *examples*.
    
    # Let's recreate exact behavior to be safe.
    
    # Re-implementing loops exactly as interpreted from original logic to avoid bugs.
    
    processed_data = {
        "inputs": [],
        "labels": [],
        "puzzle_indices": [0], # First element 0? Original: results["puzzle_indices"].append(0) before loop.
        "group_indices": [0],
        "puzzle_identifiers": []
    }
    
    p_id = 0
    e_id = 0
    
    for orig_inp, orig_out in tqdm(zip(inputs, labels), total=len(inputs)):
        # Aug loop (0 is original, >0 are shuffles)
        for i in range(1 + num_augments):
            if i == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)
            
            processed_data["inputs"].append(inp)
            processed_data["labels"].append(out)
            e_id += 1
            
            # Original: results["puzzle_indices"].append(example_id)
            # wait, original incremented example_id *before* appending? 
            # Original:
            # example_id += 1
            # puzzle_id += 1 (Wait, puzzle_id incremented inside inner loop? NO.)
            
            # Let's re-read original code carefully.
            # for orig_inp, orig_out in zip(tqdm(inputs), labels):
            #    for aug_idx in range(1 + num_augments):
            #        ...
            #        example_id += 1
            #        puzzle_id += 1  <-- Incremented INSIDE inner loop?
            #        results["puzzle_indices"].append(example_id)
            #        results["puzzle_identifiers"].append(0)
            #    results["group_indices"].append(puzzle_id)
            
            # If puzzle_id is incremented inside the inner loop, then "puzzle_id" actually counts *examples*?
            # But group_indices appends it outside.
            # So group_indices will contain [0, (1+aug), 2*(1+aug), ...]?
            # YES. If puzzle_id increments per *example* (inside inner loop), then `group_indices` tracks the cumulative count of examples.
            # That makes `group_indices` effectively the boundaries of groups in the flat array!
            # So `puzzle_id` is a misnomer in original code, it should be `cumulative_examples`.
            
    # Fix my logic to match this understanding:
    # puzzle_id is effectively running count of examples.
    
    processed_data = {
        "inputs": [],
        "labels": [],
        "puzzle_indices": [0], 
        "group_indices": [0],
        "puzzle_identifiers": []
    }
    
    current_idx = 0
    for orig_inp, orig_out in tqdm(zip(inputs, labels), total=len(inputs)):
        for i in range(1 + num_augments):
            if i == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)
            
            processed_data["inputs"].append(inp)
            processed_data["labels"].append(out)
            
            current_idx += 1
            processed_data["puzzle_indices"].append(current_idx)
            processed_data["puzzle_identifiers"].append(0)
            
        processed_data["group_indices"].append(current_idx)
        
    # Convert to numpy and save
    # Inputs/Labels are 1-9, 0 is PAD?
    # Original: _seq_to_numpy adds 1 to everything?
    # Original: arr + 1. assert >=0 and <=9.
    # Input was 0..9. So output is 1..10.
    # Metadata says vocab_size=11 (PAD + 0..9). 
    # Wait, if input has 0 (blank), and we add 1, then blank becomes 1. 
    # Sudoku digits 1-9 become 2-10.
    # PAD is usually 0.
    # So 0 is PAD. 1 is "0" (blank). 2..10 are "1".."9".
    
    def _to_numpy(seq):
        arr = np.concatenate([x.reshape(1, -1) for x in seq], axis=0) # flatten to (N, 81)
        return arr + 1 # Shift for padding
    
    final_inputs = _to_numpy(processed_data["inputs"])
    final_labels = _to_numpy(processed_data["labels"])
    
    save_path = os.path.join(args.output_dir, set_name)
    os.makedirs(save_path, exist_ok=True)
    
    np.save(os.path.join(save_path, "all__inputs.npy"), final_inputs)
    np.save(os.path.join(save_path, "all__labels.npy"), final_labels)
    np.save(os.path.join(save_path, "all__group_indices.npy"), np.array(processed_data["group_indices"], dtype=np.int32))
    np.save(os.path.join(save_path, "all__puzzle_indices.npy"), np.array(processed_data["puzzle_indices"], dtype=np.int32))
    np.save(os.path.join(save_path, "all__puzzle_identifiers.npy"), np.array(processed_data["puzzle_identifiers"], dtype=np.int32))
    
    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=11,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(processed_data["group_indices"]) - 1,
        mean_puzzle_examples= (1 + num_augments),
        total_puzzles=len(processed_data["group_indices"]) - 1,
        sets=["all"]
    )
    
    with open(os.path.join(save_path, "dataset.json"), "w") as f:
        f.write(metadata.model_dump_json())

    # Identifiers
    with open(os.path.join(args.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

def main():
    args = parse_args()
    process_subset("train", args)
    process_subset("test", args)

if __name__ == "__main__":
    main()
