
protoc -I=configuration --python_out=configuration \
  configuration/environment.proto \
  configuration/config.proto


# Stupid python protos are broken https://github.com/protocolbuffers/protobuf/issues/1491
sed -i '' 's/import environment_pb2 as environment__pb2/from . import environment_pb2 as environment__pb2/g' \
  configuration/config_pb2.py
