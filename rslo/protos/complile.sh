cd ../../
find  .  | grep  "\./rslo/protos/.*\.proto$" | xargs -n1 -I {}  protoc --proto_path . --python_out=. {}

# find ls | grep ".proto" | xargs -n1 -I {} protoc --proto_path . --python_out=. {}