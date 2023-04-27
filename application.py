from flask import Flask,render_template,jsonify,request
from src.pipeline.predict_pipline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])


def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    
    else:
        data=CustomData(
            Delivery_person_Age=float(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings=float(request.form.get('Delivery_person_Ratings')),
            Restaurant_latitude=float(request.form.get('Restaurant_latitude')),
            Restaurant_longitude=float(request.form.get('Restaurant_longitude')),
            Delivery_location_latitude=float(request.form.get('Delivery_location_latitude')),
            Delivery_location_longitude=float(request.form.get('Delivery_location_longitude')),
            Weather_conditions=request.form.get('Weather_conditions'),
            Road_traffic_density=request.form.get('Road_traffic_density'),
            Vehicle_condition=float(request.form.get('Vehicle_condition')),
            Type_of_order=request.form.get('Type_of_order'),
            Type_of_vehicle=request.form.get("Type_of_vehicle"),
            multiple_deliveries=float(request.form.get("multiple_deliveries")),
            Festival=request.form.get('Festival'),
            City=request.form.get('City'),
            order_day=float(request.form.get('order_day')),
            order_month=float(request.form.get('order_month')),
            order_hours=float(request.form.get('order_hours')),
            order_min=float(request.form.get('order_min')),
            picked_hours=float(request.form.get('picked_hours')),
            picked_min=float(request.form.get('picked_min'))

        )

        final_new_data=data.get_data_as_dataframe()
        preduct_pipeline=PredictPipeline()
        pred=preduct_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template("result.html",final_result=results)

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)








