use core::fmt;

#[derive(Debug, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    BinaryOp {
        left: Box<Expr>,
        op: Operator,
        right: Box<Expr>,
    }
}

impl Expr {
    fn fmt_with_parens(&self, fmtr: &mut fmt::Formatter<'_>, parent_prec: u8) -> fmt::Result {
        match self {
            Expr::Int(n) => write!(fmtr, "{}", n),
            Expr::Float(f) => write!(fmtr, "{}", f),
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

pub fn evaluate(expr: &Expr) -> i64 {
    match expr {
        Expr::Int(n) => *n,
        Expr::Float(f) => *f as i64,
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

pub fn evaluate_64(expr: &Expr) -> Result<f64, &'static str> {
    match expr {
        Expr::Int(n) => Ok(*n as f64),
        Expr::Float(f) => Ok(*f),
        Expr::BinaryOp {left, op, right} => {
            let l = evaluate_64(left)?;
            let r = evaluate_64(right)?;

            match op {
                Operator::Add => Ok(l + r),
                Operator::Sub => Ok(l - r),
                Operator::Mul => Ok(l * r),
                Operator::Div => {
                    if r == 0.0 {
                        Err("division by Zero")
                    }else{
                        Ok(l / r)
                    }
                }

            }

        }
    }
}

pub fn evaluate_i64(expr: &Expr) -> Result<i64, &'static str>{
    match expr {
        Expr::Int(n) => Ok(*n),
        Expr::Float(f) => Ok(*f as i64),
        Expr::BinaryOp { left, op, right } => {
            let l = evaluate_i64(left)?;
            let r = evaluate_i64(right)?;
            match op {
                Operator::Add => Ok(l + r),
                Operator::Sub => Ok(l - r),
                Operator::Mul => Ok(l * r),
                Operator::Div => {
                    if r == 0 {
                        Err("division by Zero")
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


    use crate::{add, bin, div, evaluate, evaluate_64, evaluate_i64, float, mul, num, sub, Expr, Operator};

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
        assert_eq!(result, Err("division by Zero"));
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
        assert_eq!(result, Err("division by Zero"));
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
}