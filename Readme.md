# calc_ast
Rustでの小さな構文解析

# 構成
- lib.rs：AST定義、トークナイザ（tokenize）、再帰下降パーサ（parse_exprなど）、評価器（evaluate）
- bin/repl.rs：REPLとして、入力された数式を評価

# 使い方
## test
cargo test
## repl 
cargo run --bin repl

# 残件
- 浮動小数点のtoken化
