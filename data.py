import pandas as pd

All_Num = 32768
Train_Num = int(All_Num*0.7)

File_Name = 'data/20210915/新做的/dila_conv/2048hz/100%'
response_name = '/输入.xlsx'
data = pd.read_excel(File_Name + response_name,header=None)#excel即使修改了后缀名为csv也不是csv文件，按照pd.read_csv来读取会报错,转一下数据格式
data1=data.iloc[:Train_Num,0:]
data1.to_csv(File_Name+'/response.csv', index = None, header=None, encoding='utf-8')
data2 = data.iloc[Train_Num:,0:]

data2.to_csv(File_Name+'/response-test.csv', index = None, header=None, encoding='utf-8')

stimulate_name = '/输出.xlsx'
label= pd.read_excel(File_Name + stimulate_name,header=None)
label1= label.iloc[:Train_Num,0:]
label1.to_csv(File_Name+'/stimulate.csv', index=None, header=None, encoding='utf-8')
label2 = label.iloc[Train_Num:,0:]
label2.to_csv(File_Name+'/stimulate-test.csv', index=None, header=None, encoding='utf-8')