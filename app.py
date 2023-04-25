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

        )
        final_new_data=data.get_data_as_dataframe()
        preduct_pipeline=PredictPipeline()
        pred=preduct_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template("result.html",final_result=results)

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)








