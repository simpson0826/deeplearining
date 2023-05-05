import pandas as pd
import numpy as np
df1 = pd.read_excel('./DOE/DOEsetting_MLP.xlsx', sheet_name = 'sheet1')
df2 = pd.read_excel('./DOE/DOEsetting_MLP.xlsx', sheet_name = 'sheet2')

df = pd.DataFrame(np.full((405, 10), np.nan))
df.columns = list(df1.columns)

for i in df.columns:
    for j in df.index:
        df[i][j] = df2[i][df1[i][j]]
        
df_shuffled=df.sample(frac=1).reset_index(drop=True)
# 结果输出到result.xls
#print(df_shuffled)
# 结果输出到result.xls
filepath = './DOE/DOE_result_MLP.xlsx'
writer = pd.ExcelWriter(filepath)
df_shuffled.to_excel(excel_writer=writer, index=False, sheet_name='Sheet1')
writer.save()
writer.close()
