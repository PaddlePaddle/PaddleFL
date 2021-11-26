## 联邦特征工程

支持计算正样本占比、woe、iv

## 准备工作

编译paillier(参照docs/source/md/compile_and_install_cn.md 编译PaddleFL)
```
cd ../../../build
make -j48
make install
```
完成后在feature/python/libs 目录下会有 he_utils.so

生成proto
```
cd ../feature/proto
python3 run_protogen.py
```

## 准备数据
```
cd ../python/example
python3 gen_test_file.py
```
简单测试: gen_simple_file  性能测试: gen_bench_file
## 生成证书
生成grpc证书 grpc secure channel 需要用到

```
openssl req -newkey rsa:2048 -nodes -keyout server.key -x509 -days 3650 -out server.crt
```
示例中定义Common Name 为 metrics_service 其余为空

在example目录下会生成 server.key server.crt

## 流程示例
channel: grpc client channel 可自定义

server: grpc server 可自定义
```
#client
    fed_fea_eng_client = FederalFeatureEngineeringClient(1024)
    fed_fea_eng_client.connect(channel)
    result = fed_fea_eng_client.get_woe(labels)
#server
    fed_fea_eng_server = FederalFeatureEngineeringServer()
    fed_fea_eng_server.serve(server)
    woe_list = fed_fea_eng_server.get_woe(features)
```
## 具体例子及测试
服务器端：python3 metrics_test_server.py 

客户端： python3 metrics_test_client.py
