# DeepLearning
# 딥러닝 모델 및 알고리즘 구현 프로젝트 (Deep Learning Models and Algorithms Implementation Project)

이 저장소는 다양한 딥러닝 모델과 핵심 알고리즘을 학습하고 구현한 내용을 담고 있습니다. 각 폴더는 특정 모델 또는 개념에 대한 이론적 배경과 구현 코드를 포함하고 있습니다.

## 📁 폴더 구조

DeepLearning/
│ ├─ 1. 퍼셉트론.ipynb
│ ├─ 18_NeuralNetwork.ipynb
│ ├─ 19_BackPropagation.ipynb
│ ├─ 2. 다층 퍼셉트론.ipynb
│ ├─ 20_MNIST_Data_Training.ipynb
│ ├─ 21_CNN_Model.ipynb
│ ├─ CRNN.ipynb
│ ├─ Ex_MLP_FF_starter.ipynb
│ ├─ LSTM_Stock_Market_Analysis.ipynb
│ ├─ MLP_손실_함수와_활성화_함수_시각화.ipynb
│ ├─ MLP와_XOR_문제_시각화.ipynb
│ ├─ PracticeNote.ipynb
│ ├─ RNN_LSTM.ipynb
│ ├─ RNN_Model.ipynb
│ ├─ XOR_Problem_with_sklearn_MLPClassifier.ipynb
│ ├─ XOR_Problem_with_sklearn_MLPClassifier.ipynb
│ ├─ 은닉층_XOR_신경망_시각화.ipynb
└─ README.md

## 목차

1.  [Neural Network (신경망)](#1-neuralnetwork-신경망)
2.  [Perceptron (퍼셉트론)](#2-perceptron-퍼셉트론)
3.  [MLP (Multi-Layer Perceptron, 다층 퍼셉트론)](#3-mlp-multi-layer-perceptron-다층-퍼셉트론)
4.  [BackPropagation (오차 역전파)](#4-backpropagation-오차-역전파)
5.  [MNIST Data Training (MNIST 데이터 학습)](#5-mnist_data_traning-mnist-데이터-학습)
6.  [CNN Model (Convolutional Neural Network, 합성곱 신경망)](#6-cnn_model-convolutional-neural-network-합성곱-신경망)
7.  [RNN Model (Recurrent Neural Network, 순환 신경망)](#7-rnn_model-recurrent-neural-network-순환-신경망)
8.  [RNN_LSTM (Long Short-Term Memory in RNN)](#8-rnn_lstm-long-short-term-memory-in-rnn)
9.  [CRNN (Convolutional Recurrent Neural Network)](#9-crnn-convolutional-recurrent-neural-network)
10.  [LSTM (Long Short-Term Memory)](#10-lstm-long-short-term-memory)
11. [XOR 문제 시각화 & MLP 예제](#11-xor-문제-시각화--mlp-예제)

---

## 1. NeuralNetwork (신경망)

신경망은 인간의 뇌를 구성하는 뉴런(신경 세포)의 동작 방식에 착안하여 만들어진 계산 모델입니다. 입력층(Input Layer), 은닉층(Hidden Layer), 출력층(Output Layer)으로 구성되며, 각 층은 여러 개의 노드(뉴런)를 포함합니다.

* **퍼셉트론 (Perceptron):** 신경망의 기본 단위로, 다수의 입력을 받아 가중치를 곱하고, 활성화 함수를 통해 출력을 결정합니다.
* **활성화 함수 (Activation Function):** 입력 신호의 총합을 출력 신호로 변환하는 함수입니다. 비선형성을 도입하여 신경망이 복잡한 패턴을 학습할 수 있도록 합니다.
    * 예시: Sigmoid, ReLU, Tanh 등

➡️ [18_NeuralNetwork.ipynb](./18_NeuralNetwork.ipynb)

## 2. Perceptron (퍼셉트론)
퍼셉트론은 1958년 Frank Rosenblatt이 제안한 가장 단순한 형태의 인공 신경망으로, 두 클래스(0/1)를 선형 결정 경계로 구분하는 이진 분류기(binary classifier)입니다.

➡️ [1. 퍼셉트론.ipynb](./1.%20퍼셉트론.ipynb) 

## 3. MLP (Multi-Layer Perceptron, 다층 퍼셉트론)

다층 퍼셉트론(MLP)은 입력층과 출력층 사이에 하나 이상의 은닉층을 가진 신경망입니다. 단층 퍼셉트론(SLP)이 선형 분리 가능한 문제만 해결할 수 있는 반면, MLP는 비선형 분리 가능한 문제도 해결할 수 있습니다.

* **특징:**
    * 각 층의 뉴런은 이전 층의 모든 뉴런과 연결됩니다 (Fully Connected Layer).
    * 오차 역전파 알고리즘을 사용하여 가중치를 학습합니다.
    * XOR 문제와 같이 복잡한 문제 해결에 사용될 수 있습니다.

➡️ [2. 다층 퍼셉트론.ipynb](./2.%20다층%20퍼셉트론.ipynb)
➡️ [MLP와_XOR_문제_시각화.ipynb](./MLP와_XOR_문제_시각화.ipynb)

## 4. BackPropagation (오차 역전파)

오차 역전파는 신경망 학습에서 가장 핵심적인 알고리즘 중 하나입니다. 출력층에서 발생한 오차를 입력층 방향으로 역으로 전파하면서 각 층의 가중치와 편향을 업데이트합니다.

* **원리:**
    1.  순전파 (Forward Propagation): 입력 데이터를 신경망에 통과시켜 예측값을 계산합니다.
    2.  오차 계산: 예측값과 실제값 사이의 오차를 계산합니다 (예: 평균 제곱 오차).
    3.  역전파 (Backward Propagation): 오차를 각 층으로 역전파하며, 각 가중치가 오차에 얼마나 기여했는지 (기울기) 계산합니다.
    4.  가중치 업데이트: 계산된 기울기를 사용하여 경사 하강법(Gradient Descent) 등의 최적화 알고리즘으로 가중치를 업데이트합니다.
* **목표:** 오차를 최소화하는 방향으로 신경망의 파라미터(가중치, 편향)를 조정합니다.

➡️ [19_BackPropagation.ipynb](./19_BackPropagation.ipynb)

## 5. MNIST_Data_Traning (MNIST 데이터 학습)

MNIST 데이터셋은 손으로 쓴 숫자 이미지(0부터 9까지)로 구성된 대규모 데이터베이스입니다. 딥러닝 및 머신러닝 분야에서 모델의 성능을 테스트하고 비교하는 데 널리 사용되는 표준 데이터셋 중 하나입니다.

* **구성:**
    * 60,000개의 학습 이미지
    * 10,000개의 테스트 이미지
    * 각 이미지는 28x28 픽셀의 흑백 이미지입니다.
* **활용:** 이미지 분류(Image Classification) 모델의 학습 및 평가에 주로 사용됩니다. 이 폴더에서는 MNIST 데이터셋을 사용하여 특정 모델(예: MLP, CNN)을 학습시키는 과정을 다룹니다.

➡️ [20_MNIST_Data_Training.ipynb](./20_MNIST_Data_Training.ipynb)

## 6. CNN_Model (Convolutional Neural Network, 합성곱 신경망)

합성곱 신경망(CNN)은 특히 이미지 처리 분야에서 뛰어난 성능을 보이는 딥러닝 모델입니다. 인간의 시각 처리 방식을 모방하여 설계되었습니다.

* **주요 구성 요소:**
    * **합성곱 계층 (Convolutional Layer):** 입력 데이터에 필터(커널)를 적용하여 특징 맵(Feature Map)을 추출합니다. 이미지의 지역적인 특징을 학습합니다.
    * **풀링 계층 (Pooling Layer):** 특징 맵의 크기를 줄여 계산량을 감소시키고, 주요 특징만 남깁니다. (예: Max Pooling, Average Pooling)
    * **완전 연결 계층 (Fully Connected Layer):** MLP와 유사하게, 추출된 특징들을 바탕으로 최종적인 분류 또는 예측을 수행합니다.
* **특징:**
    * 파라미터 공유(Parameter Sharing)를 통해 모델의 파라미터 수를 줄이고 효율성을 높입니다.
    * 지역적 연결성(Local Connectivity)을 통해 이미지의 공간적 구조를 잘 학습합니다.

➡️ [21_CNN_Model.ipynb](./21_CNN_Model.ipynb)

## 7. RNN_Model (Recurrent Neural Network, 순환 신경망)

순환 신경망(RNN)은 시퀀스(Sequence) 데이터 처리에 특화된 신경망입니다. 이전 시간 단계(time step)의 출력이 현재 시간 단계의 입력으로 사용되는 순환 구조를 가지고 있어, 시간적 연속성이 있는 데이터를 효과적으로 처리할 수 있습니다.

* **특징:**
    * 음성 인식, 자연어 처리, 시계열 예측 등 순서가 중요한 데이터에 주로 사용됩니다.
    * 내부적으로 '기억(memory)'을 가지는 형태로 동작합니다.
* **한계점:**
    * **장기 의존성 문제 (Long-Term Dependency Problem):** 시퀀스가 길어질수록 과거의 정보가 제대로 전달되지 못하는 문제가 발생할 수 있습니다 (기울기 소실 또는 폭주 문제).

➡️ [RNN_Model.ipynb](./RNN_Model.ipynb)

## 8. RNN_LSTM (Long Short-Term Memory in RNN)

LSTM(Long Short-Term Memory)은 RNN의 장기 의존성 문제를 해결하기 위해 고안된 발전된 형태의 순환 신경망입니다. '셀 상태(Cell State)'와 여러 '게이트(Gate)' 구조를 도입하여 정보의 흐름을 효과적으로 제어합니다.

* **주요 구성 요소:**
    * **셀 상태 (Cell State):** 정보를 장기간 기억하는 컨베이어 벨트와 같은 역할을 합니다.
    * **게이트 (Gates):**
        * **망각 게이트 (Forget Gate):** 과거 정보 중 어떤 것을 버릴지 결정합니다.
        * **입력 게이트 (Input Gate):** 새로운 정보 중 어떤 것을 셀 상태에 저장할지 결정합니다.
        * **출력 게이트 (Output Gate):** 셀 상태를 기반으로 어떤 값을 출력할지 결정합니다.
* **장점:** RNN에 비해 긴 시퀀스에서도 정보를 효과적으로 기억하고 전달할 수 있습니다.

➡️ [RNN_LSTM.ipynb](./RNN_LSTM.ipynb)


## 9. CRNN (Convolutional Recurrent Neural Network)

CRNN은 합성곱 신경망(CNN)과 순환 신경망(RNN, 주로 LSTM 또는 GRU)을 결합한 모델 구조입니다. CNN을 통해 입력 데이터(주로 이미지나 시계열 데이터의 특징)로부터 공간적 또는 지역적 특징을 추출하고, 추출된 특징 시퀀스를 RNN에 입력하여 시간적 패턴이나 순서 정보를 학습합니다.

* **활용 분야:**
    * 이미지 캡셔닝 (Image Captioning)
    * 광학 문자 인식 (Optical Character Recognition, OCR)
    * 음성 인식 (Speech Recognition)
    * 비디오 분석 (Video Analysis)
* **장점:** 공간적 특징과 시간적 특징을 동시에 효과적으로 모델링할 수 있습니다.

➡️ [CRNN.ipynb](./CRNN.ipynb)

## 10. LSTM (Long Short-Term Memory)

이 폴더는 LSTM 네트워크 자체에 대한 심층적인 이론이나 특정 구현 방식을 다룰 수 있습니다. (위의 `08.RNN_LSTM` 과의 차별점을 명확히 하여 기술하는 것이 좋습니다. 예를 들어, `08.RNN_LSTM`이 RNN 프레임워크 내에서의 LSTM 활용을 다룬다면, 이 폴더는 LSTM 유닛 자체의 다양한 변형이나 세부 메커니즘을 다룰 수 있습니다.)

* **핵심 개념:** (RNN_LSTM 섹션과 유사하지만, 더 깊이 있는 설명이나 특정 변형 모델에 대한 설명 추가 가능)
    * 셀 상태와 게이트 메커니즘의 중요성
    * LSTM의 다양한 변형 (예: Peephole LSTM, GRU 등과의 비교)
    * LSTM이 장기 의존성 문제를 해결하는 방식에 대한 상세 설명

➡️ [LSTM_Stock_Market_Analysis.ipynb](./LSTM_Stock_Market_Analysis.ipynb)

## 11. XOR 문제 시각화 & MLP 예제

- **XOR 스터디**:  
  - 순수 구현: [XOR_문제를_해결하는_신경망.ipynb](./XOR_문제를_해결하는_신경망.ipynb)  
  - Scikit-learn MLP: [XOR_Problem_with_sklearn_MLPClassifier.ipynb](./XOR_Problem_with_sklearn_MLPClassifier.ipynb)  
- **손실·활성화 함수 시각화**: [MLP_손실_함수와_활성화_함수_시각화.ipynb](./MLP_손실_함수와_활성화_함수_시각화.ipynb)  
- **은닉층 변화 시각화**: [은닉층_XOR_신경망_시각화.ipynb](./은닉층_XOR_신경망_시각화.ipynb)
- 
---

## 🚀 설치 및 실행 방법

```bash
git clone https://github.com/YourUser/DeepLearning.git
cd DeepLearning
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate.bat     # Windows
pip install -r requirements.txt
jupyter notebook

---

**참고:** 각 폴더에는 해당 이론을 바탕으로 구현된 Python 코드와 이론 설명이 포함되어 있습니다. 자세한 내용은 각 폴더의 소스 코드를 참고해주세요.

---

## License

This project is licensed under the **MIT License**.

