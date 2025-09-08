import json
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.utils.cell import coordinate_from_string, column_index_from_string
import os

# 读取配置文件中的所有模型的output.csv路径
def load_config(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    csv_files, used_items = [], []
    for items in data['items']:
        if items['model_name'] not in used_items:
            csv_files.append(data['dst_path'] + items['model_name'] + "/output.csv")
            used_items.append(items['model_name'])
    return csv_files

# 读取所有模型的output.csv文件并合并到excel文件中
def merge_csv_files(csv_files, dst_files):
    merged_df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        merged_df_list.append(df)
    merged_df = pd.concat(merged_df_list, ignore_index=True)

    merged_df.to_excel(dst_files, index=False)

# 后处理
def postprocess_excel(excel_files, dst_files):
    wb = load_workbook(excel_files)
    ws = wb.active

    column = 1
    max_row = ws.max_row

    # 将相同的模型名称合并单元格
    prev_value = None
    merge_start = 1
    for row in range(2, max_row + 1):
        cell = ws.cell(row=row, column=column)
        if cell.value == prev_value:
            continue
        else:
            if prev_value is not None:
                ws.merge_cells(start_row=merge_start, start_column=column, end_row=row - 1, end_column=column)
            prev_value = cell.value
            merge_start = row

    if prev_value is not None:
        ws.merge_cells(start_row=merge_start, start_column=column, end_row=max_row, end_column=column)

    #####################################################
    # 新增sys_info一列
    # 需要系统信息获取脚本sys_info.sh

    if os.path.exists('sysinfo.txt'):
        with open("sysinfo.txt", 'r') as f:
            sys = f.read()
    elif os.path.exists('sys_info.sh'):
        os.system("bash sys_info.sh")
        with open("sysinfo.txt", 'r') as f:
            sys = f.read()
    else:
        sys = ""

    merge_value = "sys_info"
    max_column = ws.max_column
    new_column = max_column + 1

    cell = ws.cell(row=1, column=new_column)
    cell.value = merge_value
    cell.font = Font(bold=True)
    for row in range(2, ws.max_row + 1):
        cell = ws.cell(row=row, column=new_column)
        cell.value = sys
    ws.merge_cells(start_row=2, start_column=new_column, end_row=ws.max_row, end_column=new_column)
    ###########################################################

    wb.save(dst_files)

if __name__ == '__main__':
    concat_output = "output.xlsx"       # 中间结果
    dst_output = "results.xlsx"         # 最终结果
    
    csv_files = load_config('./auto_quick_check_config.json')
    merge_csv_files(csv_files, concat_output)
    postprocess_excel(concat_output, dst_output)
