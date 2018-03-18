# disk_failure_prediction
Only for the coding test of the interview for the AIOps position of Alibaba

## Scan all the CSV files in the current directory

```python
import pandas as pd
import glob,os,sys
input_path='./'
output_fiel='pandas_union_concat.csv'
all_files=glob.glob(os.path.join(input_path,'sales_*'))
all_data_frames=[]
for file in all_files:
    data_frame=pd.read_csv(file,index_col=None)
    total_sales=pd.DataFrame([float(str(value).strip('$').replace(',','')) for value in data_frame.loc[:,'Sale Amount']]).sum()
    average_sales=pd.DataFrame([float(str(value).strip('$').replace(',','')) for value in data_frame.loc[:,'Sale Amount']]).mean()
    data={
        'filename':os.path.basename(file),
        'total_sales':total_sales,
        'average_sales':average_sales
    }
    all_data_frames.append(pd.DataFrame(data,columns=['filename','total_sales','average_sales']))
data_frame_concat=pd.concat(all_data_frames,axis=0,ignore_index=True)
data_frame_concat.to_csv(output_fiel,index=False)
```
