사용버전 python 3.9 

1. google 에서 만든 mediapipe를 이용하여 만든 프로젝트입니다.
2. 춤추는 영상을 가져와서 pose Estimate(인체추적) 을 하였습니다.
3. mediapipe 에서 사용한 기법에서는 holistic 기법을 사용하였고, holistic은 추적 대상이 춤을추거나 빠르게 움직일때 사용되는 기법입니다.
   pose 에서는 움직임이 정적으로 움직일때 사용되는 기법이며, 제가 가저온 영상은 춤추는 영상이기 때문에 맞지 않았습니다.
   단, 스쿼드 요가 등 자세를 정확하게 추정하기 위한 기법은 pose 가 더 알맞을겁니다.
   
