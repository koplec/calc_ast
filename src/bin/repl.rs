use std::io::{self, Write};

use calc_ast::*;

fn main(){
    println!("mini calc REPL. Type an expression or 'exit' to quit.");
    loop {
        print!("> ");
        io::stdout().flush().unwrap(); //すぐにprintを表示する

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err(){
            eprintln!("Failed to read input");
            continue;
        }

        let line = input.trim();
        if line == "exit"{
            break;
        }

        match tokenize(line){
            Ok(tokens) => {
                let mut iter = tokens.iter().peekable();
                match parse_expr(&mut iter) {
                    Ok(expr) => println!("= {}", evaluate(&expr)),
                    Err(e) => eprintln!("x parse error: {}" , e),
                }
            }
            Err(e) => eprintln!("x tokenize error: {}", e),
        }
    }
}