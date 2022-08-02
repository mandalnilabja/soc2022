import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error



def testgen(data, seq_len, targetcol):
    #Return array of all test samples
    batch = []
    df=data
    input_cols = [c for c in df.columns if c != targetcol]
    # extract sample using a sliding window
    for i in range(len(df), len(df)+1):
        frame = df.iloc[i-seq_len:i]
        batch.append([frame[input_cols].values, frame[targetcol][-1]])
    X, y = zip(*batch)
    return np.expand_dims(np.array(X),3), np.array(y)



# Create flask app
flask_app = Flask(__name__)
model = tf.keras.models.load_model('2dCNNpredNIFTY50m1.h5', compile=False)     
model.compile()

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/pipeline", methods = ["POST"])
def pipeline():
    
    f = request.files['data']
    X = pd.read_csv(f, index_col="Date", parse_dates=True)
      
    # get the names of columns
    cols = X.columns
        
    # Save the target variable as a column in dataframe and drop NaN values
    X["Target"] = (X["Close"].pct_change().shift(-1) > 0).astype(int)
    X.dropna(inplace=True)
            
    # Fit and scale into transformed dataframe
    scaler = StandardScaler().fit(X.loc[:, cols])
    X[cols] = scaler.transform(X[cols])
    data = X

    data['RoM3']=data.Close.rolling(3).mean() #Rolling Mean Closing Values of 3 days
    data['RoM5']=data.Close.rolling(5).mean() #Rolling Mean Closing Values of 5 days
    data['RoM15']=data.Close.rolling(15).mean() #Rolling Mean Closing Values of 15 days
    data['RoM30']=data.Close.rolling(30).mean() #Rolling Mean Closing Values of 30 days
    data['Vol_Diff'] = data["Volume"].pct_change().shift(-1) #Daily Volume Difference
    data['Close_Diff'] = data["Close"].pct_change().shift(-7) #Weekly Closing Value Difference

    X=X[X.Volume>0]

    #Dropping NaN rows
    data.dropna(inplace=True)        
        
    # Prepare test data
    test_data, test_target = testgen(data, 60, "Target")
        
    # Test the model
    test_out = model.predict(test_data)
    test_pred = (test_out > 0.5).astype(int)
    if test_pred==[[1]]:
        output='Up'
    else:
        output='Down'

    return render_template("index.html", prediction_text = "The predicted movement for NIFTY50 for next day is {}".format(output))

if __name__ == "__main__":
    flask_app.run(debug=True)