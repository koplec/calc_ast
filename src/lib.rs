use core::fmt;
use std::{iter::Peekable};
use std::slice::Iter;

#[derive(Debug, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    UnaryOp {
        op: UnaryOperator, 
        // Box<Expr>はヒープに置いたExprを指すポインタ
        // Exprは再帰的な構造になっている
        // 再帰構造のサイズはコンパイル時には決まらない
        // Box<Expr>はポインタのサイズで一定(8バイト)
        // これはら、コンパイル時にサイズが一定になる
        expr: Box<Expr> 
    },
    BinaryOp {
        left: Box<Expr>,
        op: Operator,
        right: Box<Expr>,
    }
}

#[derive(Debug, PartialEq)]
pub enum UnaryOperator{
    Neg,
}



impl Expr {
    fn fmt_with_parens(&self, fmtr: &mut fmt::Formatter<'_>, parent_prec: u8) -> fmt::Result {
        match self {
            Expr::Int(n) => write!(fmtr, "{}", n),
            Expr::Float(f) => write!(fmtr, "{}", f),
            Expr::UnaryOp { op, expr } => {
                match op {
                    UnaryOperator::Neg => {
                        write!(fmtr, "-")?;
                        expr.fmt_with_parens(fmtr, 10)
                    }
                }
            },
            Expr::BinaryOp { left, op, right } => {
                let my_prec = op.precedence();
                // 親演算子よりも優先度が低いときは、かっこが必要
                let needs_paren = my_prec < parent_prec;
                if needs_paren {
                    write!(fmtr, "(")?;
                }

                left.fmt_with_parens(fmtr, my_prec)?;
                write!(fmtr, " {} ", op)?;
                right.fmt_with_parens(fmtr, my_prec)?;

                if needs_paren {
                    write!(fmtr, ")")?;
                }

                Ok(())
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div
}

impl Operator {
    fn precedence(&self) -> u8 {
        match self{
            Operator::Add | Operator::Sub => 1,
            Operator::Div | Operator::Mul => 2,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_with_parens(f, 0)
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let symbol = match self {
            Operator::Add => "+",
            Operator::Div => "/",
            Operator::Mul => "*",
            Operator::Sub => "-",
        };
        write!(f, "{}", symbol)
    }
}

#[derive(Debug, PartialEq)]
pub enum Token {
    Int(i64),
    Plus,
    Minus,
    Star,
    Slash,
    LParen, //(
    RParen, //)
}

pub fn tokenize(input: &str) -> Result<Vec<Token>, &'static str> {
    let mut tokens = Vec::new();
    //文字列を１文字ずつ取り出す
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        if ch.is_ascii_whitespace(){
            chars.next();
        }else if ch.is_ascii_digit() {
            let mut num_str = String::new();
            while let Some(&digit) = chars.peek() {
                if digit.is_ascii_digit() {
                    num_str.push(digit);
                    chars.next();
                }else{
                    break;
                }
            }
            let num = num_str.parse::<i64>().map_err(|_| "Failed to parse integer")?;
            tokens.push(Token::Int(num));
        }else{
            match ch {
                '+' => {tokens.push(Token::Plus); chars.next();},
                '-' => {tokens.push(Token::Minus); chars.next();},
                '*' => {tokens.push(Token::Star); chars.next();},
                '/' => {tokens.push(Token::Slash); chars.next();},
                '(' => {tokens.push(Token::LParen); chars.next();},
                ')' => {tokens.push(Token::RParen); chars.next();}
                _ => return Err("Unknown character"),
            }
        }
    }

    Ok(tokens)
}

// +, - でつなげていく
pub fn parse_expr(tokens: &mut Peekable<Iter<Token>>) -> Result<Expr, &'static str> {
    let mut left = parse_term(tokens)?; //左でtermとして評価するところまでtokensを消費
   
    while let Some(token) = tokens.peek() { //現在のtokenの次もきちんとtokenがある間
        match token {
            Token::Plus | Token::Minus => { 
                let op = match tokens.next().unwrap() {
                    Token::Plus => Operator::Add,
                    Token::Minus => Operator::Sub,
                    _ => unreachable!(),
                };
                let right = parse_term(tokens)?;
                left = bin(left, op, right);
            },
            _ => break,
        }
    }
    Ok(left)
}

// *, / の演算
pub fn parse_term(tokens: &mut Peekable<Iter<Token>>) -> Result<Expr, &'static str> {
    let mut left = parse_factor(tokens)?;

    while let Some(token) = tokens.peek() {
        match token {
            Token::Star | Token::Slash => {
                let op = match tokens.next().unwrap() {
                    Token::Star => Operator::Mul,
                    Token::Slash => Operator::Div,
                    _ => unreachable!(),
                };
                let right = parse_factor(tokens)?;
                left = bin(left, op, right);
            },
            _ => break,
        }
    }
    Ok(left)
}

pub fn parse_factor(tokens: &mut Peekable<Iter<Token>>) -> Result<Expr, &'static str> {
    match tokens.next() {
        Some(Token::Int(n)) => Ok(num(*n)), //tokensでiterのときに&Tokenなので、Token::Int(n)のnが&i64になる
        Some(Token::Minus) => {
            let expr = parse_factor(tokens)?;
            Ok(Expr::UnaryOp { op: UnaryOperator::Neg, expr: Box::new(expr) })
        }
        Some(Token::LParen) => {
            let expr = parse_expr(tokens)?; //左かっこの中身を再帰的に読み取る
            match tokens.next() {
                Some(Token::RParen) => Ok(expr),
                _ => Err("Expected closing parenthesis"),
            }
        }
        _ => Err("Expected integer or opening parenthesis"),
    }
}

#[derive(Debug, PartialEq)]
pub enum EvalError {
    DivisionByZero,
    FloatInI64
}

pub fn evaluate(expr: &Expr) -> i64 {
    match expr {
        Expr::Int(n) => *n,
        Expr::Float(f) => *f as i64,
        Expr::UnaryOp { op, expr } => {
            let val = evaluate(expr);
            match op {
                UnaryOperator::Neg => -val,
            }
        },
        Expr::BinaryOp { left, op, right } => {
            let l = evaluate(left);
            let r = evaluate(right);
            match op {
                Operator::Add => l + r,
                Operator::Sub => l - r,
                Operator::Mul => l * r,
                Operator::Div => l / r,
            }
        }
    }
}

pub fn evaluate_64(expr: &Expr) -> Result<f64, EvalError> {
    match expr {
        Expr::Int(n) => Ok(*n as f64),
        Expr::Float(f) => Ok(*f),
        Expr::UnaryOp { op, expr } => {
            let val = evaluate_64(expr)?;
            match op {
                UnaryOperator::Neg => Ok(-val)
            }
        },
        Expr::BinaryOp {left, op, right} => {
            let l = evaluate_64(left)?;
            let r = evaluate_64(right)?;

            match op {
                Operator::Add => Ok(l + r),
                Operator::Sub => Ok(l - r),
                Operator::Mul => Ok(l * r),
                Operator::Div => {
                    if r == 0.0 {
                        Err(EvalError::DivisionByZero)
                    }else{
                        Ok(l / r)
                    }
                }

            }

        }
    }
}

pub fn evaluate_i64(expr: &Expr) -> Result<i64, EvalError>{
    match expr {
        Expr::Int(n) => Ok(*n),
        Expr::Float(f) => Err(EvalError::FloatInI64),
        Expr::UnaryOp { op, expr } => {
            // exprは&Box<Expr>だけれどevaluate_i64の第１引数に入れられる
            // これはRustのDefer強制という仕組みがあって、Box<Expr>を使う場面で
            // &Exprが必要であれば、自動的に参照を解決してくれる仕組み
            let val = evaluate_i64(expr)?;
            match op {
                UnaryOperator::Neg => Ok(-val)
            }
        },
        Expr::BinaryOp { left, op, right } => {
            let l = evaluate_i64(left)?;
            let r = evaluate_i64(right)?;
            match op {
                Operator::Add => Ok(l + r),
                Operator::Sub => Ok(l - r),
                Operator::Mul => Ok(l * r),
                Operator::Div => {
                    if r == 0 {
                        Err(EvalError::DivisionByZero)
                    }else{
                        Ok(l / r)
                    }
                }

            }

        }
    }
}

pub fn num(n: i64) -> Expr {
    Expr::Int(n)
}
pub fn float(x: f64) -> Expr {
    Expr::Float(x)
}
pub fn bin(left: Expr, op: Operator, right: Expr) -> Expr {
    Expr::BinaryOp { left: Box::new(left), op: op, right: Box::new(right) }
}

pub fn add(l: Expr, r: Expr) -> Expr{
    bin(l, Operator::Add, r)
}
pub fn sub(l : Expr, r: Expr) -> Expr{
    bin(l, Operator::Sub, r)
}
pub fn mul(l: Expr, r: Expr) -> Expr{
    bin(l, Operator::Mul, r)
}
pub fn div(l: Expr, r: Expr) -> Expr{
    bin(l, Operator::Div, r)
}

#[cfg(test)]
mod tests{


    use crate::{add, bin, div, evaluate, evaluate_64, evaluate_i64, float, mul, num, parse_expr, parse_factor, sub, tokenize, EvalError, Expr, Operator, Token};

    #[test]
    fn test_expr_structure(){
        let expr = Expr::Int(42);
        println!("{:?}", expr);
    }

    #[test]
    fn test_parial_eq_binary_op(){
        let a = Expr::BinaryOp { 
            left: Box::new(Expr::Int(1)), 
            op: crate::Operator::Add, 
            right: Box::new(Expr::Int(2)), 
        };

        let b = Expr::BinaryOp { 
            left: Box::new(Expr::Int(1)), 
            op: crate::Operator::Add, 
            right: Box::new(Expr::Int(2)), 
        };

        assert_eq!(a, b);
    }

    #[test]
    fn test_evaluate_nested_expr() {
        let expr = Expr::BinaryOp { 
            left: Box::new(Expr::BinaryOp { 
                left: Box::new(Expr::Int(1)), 
                op: Operator::Add,
                right: Box::new(Expr::Int(2)) 
            }), 
            op: Operator::Mul, 
            right: Box::new(Expr::Int(4)),
        };
        assert_eq!(evaluate(&expr), 12);
    }

    #[test]
    fn test_debug_ast_output(){
        let expr = Expr::BinaryOp { 
            left: Box::new(Expr::BinaryOp {
                left: Box::new(Expr::Int(1)), 
                op: Operator::Add, 
                right: Box::new(Expr::Int(2)) 
            }),
            op: Operator::Mul,
            right: Box::new(Expr::Int(3)),
        };

        dbg!(&expr);

        assert_eq!(evaluate(&expr), 9);

    }

    #[test]
    fn test_evaluate_with_helpers(){
        let expr = bin(
            bin(num(1), Operator::Add, num(2)),
            Operator::Mul,
            num(3)
        );
        assert_eq!(evaluate(&expr), 9);
    }

    #[test]
    fn test_evaluate_add(){
        let expr = add(num(1), num(10));
        assert_eq!(evaluate(&expr), 11);
    }

    #[test]
    fn test_evaluate_sub(){
        let expr = sub(num(43), num(2));
        assert_eq!(evaluate(&expr), 41);
    }

    #[test]
    fn test_evaluate_mul(){
        let expr = mul(num(32), num(4));
        assert_eq!(evaluate(&expr), 128);
    }

    #[test]
    fn test_evaluate_div(){
        let expr = div(num(10), num(2));
        assert_eq!(evaluate(&expr), 5);
    }

    #[test]
    fn test_evaluate_f64_basic(){
        let expr = add(float(1.5), num(2));
        let result = evaluate_64(&expr);
        assert_eq!(Ok(3.5), result);
    }

    #[test]
    fn test_evaluate_64_div(){
        let expr = div(num(3), float(2.0));
        let result = evaluate_64(&expr);
        assert_eq!(Ok(1.5), result);
    }

    #[test]
    fn test_evaluate_64_nexted_div(){
        let expr = div(num(3), div( float(2.0), float(3.0)));
        let result = evaluate_64(&expr);
        assert_eq!(Ok(4.5), result);
    }

    #[test]
    fn test_evaluate_f64_div_by_zero(){
        let expr = div(num(5), float(0.0));
        let result = evaluate_64(&expr);
        assert_eq!(result, Err(EvalError::DivisionByZero));
    }

    #[test]
    fn test_evaluate_i64_success(){
        let expr = div(add(num(4), num(2)), num(2));
        assert_eq!(evaluate_i64(&expr), Ok(3));
    }

    #[test]
    fn test_evaluate_i64_division_by_zero(){
        let expr = div(num(5), num(0));
        let result = evaluate_i64(&expr);
        assert_eq!(result, Err(EvalError::DivisionByZero));
    }

    #[test]
    fn test_expr_display(){
        let expr = mul(add(num(1), num(2)), num(3));
        let formatted = format!("{}", expr);
        assert_eq!(formatted, "(1 + 2) * 3");

        let expr = add(mul(num(1), num(2)), num(3));
        assert_eq!(format!("{}", expr), "1 * 2 + 3");

        let expr = add(num(1), add(num(2), num(3)));
        assert_eq!(format!("{}", expr), "1 + 2 + 3");
        let expr = add(add(num(1), num(4)), num(5));
        assert_eq!(format!("{}", expr), "1 + 4 + 5");
    }


    #[test]
    fn test_parse_expr(){
        let tokens = tokenize("1 + 2").expect("Failed to parse");
        let mut iter = tokens.iter().peekable(); 
        //このpeekableってない？
        // 次の要素を「見てから進むかどうか決められる」ようになる特別なイテレータ
        let parsed = parse_expr(&mut iter).expect("Failed to parse 1 + 2");
        // peekableは内部状態を持っていて、peek()やnext()を呼ぶと内部のポインタが進む
        // &mutをつけると箱の中身を出したり戻したりできる
        // &mutは変数iterを可変参照(mutable reference)として借りる記法
        // let mutとしておかないと可変参照を宣言できない

        let expected = add(num(1), num(2));
        assert_eq!(parsed, expected);

        let tokens = tokenize("1 - 2").expect("Failed to parse");
        let mut iter = tokens.iter().peekable();

        let parsed = parse_expr(&mut iter).expect("Failed to parse 1 - 2");
        let expected = sub(num(1), num(2));
        assert_eq!(parsed, expected);


        let tokens = tokenize("1 * 2").expect("Failed to parse");
        let mut iter = tokens.iter().peekable();
        let parsed = parse_expr(&mut iter).expect("Failed to parse 1 * 2");
        let expected = mul(num(1), num(2));
        assert_eq!(parsed, expected);
        
        let tokens = tokenize("1 / 2").expect("Failed to token 1 / 2");
        let mut iter = tokens.iter().peekable();

        let parsed = parse_expr(&mut iter).expect("Failed to parse 1 / 2");
        let expected = div(num(1), num(2));
        assert_eq!(parsed, expected);
    }

    #[test]
    fn test_parse_expr_invalid_operator(){
        let res = tokenize("1 ? 2");
        assert_eq!(res, Err("Unknown character"))
    }

    #[test]
    fn test_parse_expr_invalid_format(){
        let tokens = tokenize("1+").expect("failed to tokenize 1 +");
        let mut iter = tokens.iter().peekable();
        let parsed = parse_expr(&mut iter);
        assert_eq!(parsed, Err("Expected integer or opening parenthesis"));
    }

    #[test]
    fn test_parse_expr_with_parens(){
        let tokens = tokenize("1 * (2 + 3)").expect("tokenize failed 1 * (2 + 3)");
        let mut iter = tokens.iter().peekable();
        let parsed = parse_expr(&mut iter).expect("parse failed 1 * (2 + 3)");

        let expected = mul(num(1), add(num(2), num(3)));
        assert_eq!(parsed, expected);
    }

    #[test]
    fn test_tokenize_simple(){
        use crate::Token::*;
        let tokens = tokenize("1 + 2 * 3 / 4 - 5").expect("tokenize failed");
        let expected = vec![
            Int(1),
            Plus,
            Int(2),
            Star,
            Int(3),
            Slash,
            Int(4),
            Minus, 
            Int(5)
        ];
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_tokenize_paren_case(){
        use crate::Token::*;
        let tokens = tokenize("1*(2+3)").expect("tokenize failed 1*(2_3)");
        let expected = vec![
            Int(1),
            Star,
            LParen,
            Int(2),
            Plus,
            Int(3),
            RParen,
        ];
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_tokenize_invalid_char(){
        let res = tokenize("1 + 2 ? 3");
        assert_eq!(res, Err("Unknown character"));
    }


    #[test]
    fn test_parse_factor_integer(){
        let tokens = vec![Token::Int(42)];
        let mut iter = tokens.iter().peekable();
        let parsed = parse_factor(&mut iter).expect("should parse int 42");
        assert_eq!(parsed, num(42))
    }

    #[test]
    fn test_parse_factor_with_parens(){
        let tokens = vec![Token::LParen, Token::Int(1), Token::Plus, Token::Int(2), Token::RParen];
        let mut iter = tokens.iter().peekable();
        let parsed = parse_factor(&mut iter).expect("should parse parens");
        let expected = add(num(1), num(2));
        assert_eq!(parsed, expected);
    }

    #[test]
    fn test_parse_factor_missing_rparen(){
        let tokens = vec![Token::LParen, Token::Int(43), Token::Minus, Token::Int(2)];
        let mut iter = tokens.iter().peekable();
        let parsed = parse_factor(&mut iter);
        assert_eq!(parsed, Err("Expected closing parenthesis"));
    }

    #[test]
    fn test_parse_factor_invalid_token(){
        let tokens = vec![Token::Plus];
        let mut iter = tokens.iter().peekable();
        let parsed = parse_factor(&mut iter);
        assert_eq!(parsed, Err("Expected integer or opening parenthesis"))
    }

    #[test]
    fn test_parse_unary_minus(){
        let tokens = tokenize("-42").expect("tokenize failed -42");
        let mut iter = tokens.iter().peekable();
        let parsed = parse_expr(&mut iter).expect("parse failed -42");
        let expected = Expr::UnaryOp { op: crate::UnaryOperator::Neg, expr: Box::new(num(42)) };
        assert_eq!(parsed, expected);
    }

    #[test]
    fn test_parse_minus_equation(){
        let tokens = tokenize("-(2+3)").expect("tokenize -(2+3) failed");
        let mut iter = tokens.iter().peekable();
        let parsed = parse_expr(&mut iter).expect("parse -(2+3) failed");
        let expected = Expr::UnaryOp {
            op: crate::UnaryOperator::Neg,
            expr: Box::new(add(num(2), num(3)))
        };
        assert_eq!(parsed, expected);
    }

    #[test]
    fn test_parse_minus_equation2(){
        let tokens = tokenize("-4*2")
            .expect("tokenize -4*2 failed");
        let mut iter = tokens.iter().peekable();
        let parsed = parse_expr(&mut iter)
            .expect("parse -4*2 failed");
        let expected = mul(
            Expr::UnaryOp {
                op: crate::UnaryOperator::Neg,
                expr: Box::new(num(4))
            },
            num(2)
        );
        assert_eq!(parsed, expected);
    }
}