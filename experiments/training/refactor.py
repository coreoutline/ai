import os

def slice_file(path, del_intervals, import_str):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    to_delete = set()
    for s, e in del_intervals:
        for i in range(s, e+1):
            to_delete.add(i)
    
    new_lines = []
    for i in range(1, len(lines)+1):
        if i not in to_delete:
            new_lines.append(lines[i-1])
            
    # Insert safely at line 4 (after sys.path)
    new_lines.insert(3, import_str)
            
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Successfully processed {path}")

# train_qwen.py
import_qwen = "from src.core import create_coreoutline_qwen_model, CoreOutlineConfig\nfrom .utils import generate, text_to_token_ids, token_ids_to_text\n"
slice_file('c:\\Users\\tsuma.thomas\\Documents\\CoreOutline\\transformer\\experiments\\training\\train_qwen.py', [(47, 742)], import_qwen)

# fine_tune_instruct.py
import_fi = "from src.core import create_coreoutline_qwen_model, CoreOutlineConfig\nfrom .utils import generate, formatData, format_input\n"
slice_file('c:\\Users\\tsuma.thomas\\Documents\\CoreOutline\\transformer\\experiments\\training\\fine_tune_instruct.py', [(43, 730), (758, 779), (873, 903)], import_fi)

# fine_tune_instruct_ddp_optimized.py
import_fido = "from .utils import setup_distributed, cleanup_distributed, format_data, format_input, InstructionDatasetDDP as InstructionDataset\n"
slice_file('c:\\Users\\tsuma.thomas\\Documents\\CoreOutline\\transformer\\experiments\\training\\fine_tune_instruct_ddp_optimized.py', [(40, 107)], import_fido)

# fine_tune_instruct_ddp.py
import_fid = "from src.core import create_coreoutline_qwen_model, CoreOutlineConfig\nfrom .utils import generate, setup_distributed, cleanup_distributed, formatData, format_data, format_input, InstructionDatasetDDP as InstructionDataset\n"
slice_file('c:\\Users\\tsuma.thomas\\Documents\\CoreOutline\\transformer\\experiments\\training\\fine_tune_instruct_ddp.py', [(89, 156), (448, 1136), (1166, 1188), (1303, 1333)], import_fid)
