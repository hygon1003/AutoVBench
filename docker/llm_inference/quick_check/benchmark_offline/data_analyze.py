import json
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils.cell import coordinate_from_string, column_index_from_string
import os

def load_excel(excel_file):
    wb = load_workbook(excel_file)
    ws = wb.active

    row_as_arrays = []
    for row in ws.iter_rows():
        row_as_array = []
        for cell in row:
            value = None
            for merged_range in ws.merged_cells.ranges:
                if (cell.row == merged_range.min_row and cell.column >= merged_range.min_col and
                    cell.column <= merged_range.max_col) or (cell.row >= merged_range.min_row and
                                                            cell.row <= merged_range.max_row and
                                                            cell.column == merged_range.min_col):
                    value = ws.cell(merged_range.min_row, merged_range.min_col).value
                    break
            if value is None:
                value = cell.value
            row_as_array.append(value)
        row_as_arrays.append(row_as_array[:-1])
    return row_as_arrays


def analyse(new_file, old_file):
    datas_new = load_excel(new_file)[1:]
    datas_old = load_excel(old_file)[1:]

    wb = Workbook()
    ws = wb.active
    ws.title = "V.S."

    data = ['model_name', 'DCU', 'DCU num', 'precision', 'input_len', 'output_len', 'bs', \
            'TTFT_mean(s)-new', 'TTFT_mean(s)-old', 'TTFT_mean(s)-delta', \
            'TPOT_mean(s)-new', 'TPOT_mean(s)-old', 'TPOT_mean(s)-delta', \
            'GenerateThroughput(tokens/s)-new', 'GenerateThroughput(tokens/s)-old', 'GenerateThroughput(tokens/s)-delta', 
            'total_time(s)-new', 'total_time(s)-old', 'total_time(s)-delta', \
            'version-new', 'version-old']
    ws.append(data)

    for item in datas_new:
        cond = item[:7]
        for item2 in datas_old:
            if cond == item2[:7]:
                added = cond
                for i in range(7, 11):
                    added.append(item[i])
                    added.append(item2[i])
                    if item[i] == None and item2[i] == None:
                        added.append(item[i])
                    else:
                        added.append(item[i] - item2[i])
                added.append(item[11])
                added.append(item2[11])
                ws.append(added)

    for column in 10,13,19:      # TTFT、TPOT、time
        for row in ws.iter_rows(min_col=column, max_col=column, min_row=2): 
            for cell in row:
                if cell.value == None:
                    continue
                if cell.value >= 0:
                    cell.fill = PatternFill(start_color='00FF00', end_color='00FF00', fill_type='solid')
                else:
                    cell.fill = PatternFill(start_color='FF0000', end_color='FF0000', fill_type='solid')

    column = 16         # Generative Throughput
    for row in ws.iter_rows(min_col=column, max_col=column, min_row=2): 
        for cell in row:
            if cell.value == None:
                continue
            if cell.value >= 0:
                cell.fill = PatternFill(start_color='FF0000', end_color='FF0000', fill_type='solid')
            else:
                cell.fill = PatternFill(start_color='00FF00', end_color='00FF00', fill_type='solid')

    # 保存工作簿到文件
    wb.save("analyze.xlsx")

if __name__ == '__main__':
    new_results = "results_062.xlsx"            # 新版本的测试结果
    old_results = "results_050.xlsx"            # 旧版本的测试结果
    
    analyse(new_results, old_results)


