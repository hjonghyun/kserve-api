# Kserve-API Server



[Kserve pyhton SDK](https://kserve.github.io/website/0.7/sdk_docs/sdk_doc/)를 [python Flask WebFrameWork](https://flask.palletsprojects.com/en/0.12.x/)를 통해 REST-API 호출가능한 형태로 만든 서버.



## 프로젝트 구조

- /app               실제 구동할 어플리케이션 소스, 환경설정 config 포함

  /Dockerfile    도커파일

  /gunicorn.sh WSGI지원 Web서버 [gunicorn](https://gunicorn.org/) 실행 쉘 스크립트,    



## 실행 환경설정
프로덕션 환경(K8S)배포 실행일 경우 
KSERVE_API_K8S_CONFIG_FILE 를 공백으로 하면 K8S POD자체의 Service Account의 권한에 따라 실행

```
#inference service 배포할 K8S Cluster의 config file위치 
export KSERVE_API_K8S_CONFIG_FILE="~/.kube/config" 
#모델이 저장된 storage 인증정보 
export KSERVE_API_DEFAULT_STORAGE_CREDENTIAL="[default] \n AWS_ACCESS_KEY_ID:bWluaW9hZG1pbg== \n AWS_SECRET_ACCESS_KEY:bWluaW9hZG1pbg==" 
#모델이 저장된 storage 정보 기본값으로 내부 스토리지(minio) 설정됨          
export KSERVE_API_DEFAULT_STORAGE_ENDPOINT="192.168.88.154:9000"   
#inference service의 logger정보, Request/Response 정보 이벤트를 전달할 서비스위치                                                                                 
export KSERVE_API_LOGGERURL="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker"    
```

## 실행

```
./gunicorn.sh
```



## REST-API

- For swagger document you have to request root directory(/)

```
http://yourhostname:5000/
```



## Docker
- 개발환경 로컬 도커 실행 예
  -e 옵션: 개발환경 env설정
  -v 옵션: 구동에 필요한 K8S config를 컨테이너 환경으로 복사
```
docker run -e \
KSERVE_API_K8S_CONFIG_FILE="/.kube/config" \
KSERVE_API_DEFAULT_STORAGE_CREDENTIAL="[default] \n AWS_ACCESS_KEY_ID:bWluaW9hZG1pbg== \n AWS_SECRET_ACCESS_KEY:bWluaW9hZG1pbg==" \
export KSERVE_API_DEFAULT_STORAGE_ENDPOINT="192.168.88.154:9000" \
export KSERVE_API_LOGGERURL="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker" \
 -v ~/.kube/config:/.kube/config 192.168.88.155/koreserve/kserve-api:v1.0
```

