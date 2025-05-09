use calc_ast::{add, div, evaluate, evaluate_64, evaluate_i64, float, mul, num};


fn main(){
    let expr = mul(add(num(1), num(2)), num(3));
    println!("expr: {}", expr);
    println!("evaluate: {}", evaluate(&expr));
    println!("evaluate_i64:{:?}", evaluate_i64(&expr));
    println!("evaluate_64:{:?}", evaluate_64(&expr));


    let expr_float = add(float(1.2), num(3));
    println!("expr_float: {}", expr_float);
    println!("evaluate: {}", evaluate(&expr_float));
    println!("evaluate_i64:{:?}", evaluate_i64(&expr_float));
    println!("evaluate_64:{:?}", evaluate_64(&expr_float));

    let expr_zero_divide = div(num(45), num(0));
    println!("expr_zero_divide: {}", expr_zero_divide);
    // println!("evaluate: {}", evaluate(&expr_zero_divide));
    println!("evaluate_i64:{:?}", evaluate_i64(&expr_zero_divide));
    println!("evaluate_64:{:?}", evaluate_64(&expr_zero_divide));


    
}