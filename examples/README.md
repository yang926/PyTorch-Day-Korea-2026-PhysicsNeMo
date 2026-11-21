# 실습 예제 코드

| 파일 | 노트북에서 하는 일 |
| --- | --- |
| [pinn_plot.py](pinn_plot.py) | [순방향 PINN](../01_PINN.ipynb#lab-11)에서 줄의 변위 예측값과 해석해를 겹쳐 그리고, 위치별 오차를 확인합니다. |
| [fno.py](fno.py) | [FNO](../02_FNO.ipynb)에서 학습·검증·시험 데이터를 각각 `TensorDataset`으로 묶고 2차원 FNO를 만듭니다. |
| [plate_prediction.py](plate_prediction.py) | 학습한 FNO로 히터 위치와 전력에 따른 판의 온도를 예측하고 수치해와 비교합니다. |
