1. FreiHAND 현재 frame pose에서 fake pose 생성
- augmentation weight로 noise 적용. px은 0~5, depth는 0~0.005 (?) 다시 확인.
- noise 적용한 pose를 64*64 heatmap에 depth value 넣고 사방 2 pixel, 4 pixel 간격으로 /2, /3, /5 value 넣기

4. base
- lr 말고 decaying rate가 0.8? 어떤 value가 0.8이었는지 체크
- lr scheduler 주기 그대로했나?
- tensorboard로 loss visualization

3. network model
- prev pose feature + weight 받아서 supervision하는 encoder (output : 32*32*32)
- front/back에 따라 결합 위치 다르게 해서 적용
----------------


2. HO3D dataset 미리 pkl 만들어두고 load하는 방식(2021_Handtracker 참조)
- image crop 포함.
- rgb value range 체크. rgb order 체크(동일한 code 위치에서 ~ normalize, tensor 전환 전) value는 0~1로
- dataset normalization 한번 하는지 체크
- mesh 정보까지 추출(manopth로 바꾸고 했었나?)


5. Frei/HO3D evaluation code
- HO3D에는 MKA 구현
- output axis 체크

6. cross-training code
- dataloader 2개 선언한뒤 batch마다 번갈아서 load하도록 구글링
- dataset size차이때문에 1:3 간격으로 batch 제공
