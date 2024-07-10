from tensorflow.keras.models import load_model
import pandas as pd


## header ## 
##  day	month	year	Julian_Day	Julian_Century	Geom_Mean_Long_Sun_deg	Geom_Mean_Anom_Sun_deg	Eccent_Earth_Orbit	Sun_Eq_of_Ctr	Sun_True_Long_deg	Sun_True_Anom_deg	Sun_Rad_Vector_AUs	Sun_App_Long_deg	Mean_Obliq_Ecliptic_deg	Obliq_Corr_deg	Sun_Rt_Ascen_deg	Sun_Declin_deg	var_y	Eq_of_Time_minutes	HA_Sunrise_deg	Solar_Noon_LST	Sunrise_Time_LST	Sunset_Time_LST	Sunlight_Duration_minutes	True_Solar_Time_min	Hour_Angle_deg	Solar_Zenith_Angle_deg	Solar_Elevation_Angle_deg	Approx_Atmospheric_Refraction_deg	ES	YS	SX	SY	SZ	angel_x	angel_y	angel_z

path_model_image_x = "./model/model_27_jun_2024_imageX_2553_0.4160.h5" 
path_model_image_y = "./model/model_27_jun_2024_imageY_07_2.0817.h5"
path_data = "./data/data.xlsx" ## ชื่อข้อมูล
sheet_xlsx_name = "260424" ## ชื่อ Excel sheet

path_save_predict_result_file = "./predict"

def processing_data(path_data, sheet_name):
    print("start processing data...")
    raw_df = pd.read_excel(path_data, sheet_name=sheet_name)
    df = raw_df.drop(columns=['Julian_Day', 'Julian_Century'])
    return df

def predict_data(path_x, path_y, data):
    print("start predict data...")
    model_x = load_model(path_x)
    model_y = load_model(path_y)
    result_imageX = model_x.predict(data)
    result_imageY = model_y.predict(data)
    return result_imageX, result_imageY

def concat_data(predict_x, predict_y, df):
    print("start concat data...")
    df = df['predict_x'] = predict_x
    df = df['predict_y'] = predict_y
    return df

def convert_to_xlxs(path_save, data):
    print("start convert data to xlsx...")
    data.to_excel(path_save, index=False)
    
def main():
    df = processing_data(path_data,  sheet_xlsx_name)
    result_x, result_y = predict_data(path_model_image_x, path_model_image_y, df)
    df_predict = concat_data(result_x, result_y, df)
    convert_to_xlxs(path_save_predict_result_file,df_predict)
    print("end")
    
main()