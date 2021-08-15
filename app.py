# API 호출을 통해 iris 데이터 입력받고
# 모델(iris.pkl)을 통해 예측값을 계산해 반환함

from flask import Flask, render_template, request
import pickle
import numpy as np

# Flask(__name__)을 호출해서 플라스크 앱 초기화
# pickle로 저장했던 모델을 불러옴

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)

# 플라스크 앱의 루트 디렉토리를 초기화한 후, 이 루트 디렉토리에서
# 자신을 호출할 함수 (main)을 정의

@app.route('/')
def main():
    return render_template('home.html')

# request 사용해서 html 데려오고, model.predict()을 통해 내보내는 함수 정의

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

# 예측값에 따라 어떤 텍스트와 이미지를 내보낼지가 after.html에 설정


if __name__ == "__main__":
    app.run(debug=True)















