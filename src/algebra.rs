pub fn matrix_multiplication(matrix1: &Vec<Vec<f64>>, matrix2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows1 = matrix1.len();
    let cols1 = matrix1[0].len();
    let rows2 = matrix2.len();
    let cols2 = matrix2[0].len();

    if cols1 != rows2 {
        panic!("Cannot perform matrix multiplication: incompatible dimensions");
    }

    let mut result = vec![vec![0.0; cols2]; rows1];

    for i in 0..rows1 {
        for j in 0..cols2 {
            for k in 0..cols1 {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    result
}


pub fn dot_product(matrix: &Vec<Vec<f64>>, vector: &Vec<f64>) -> Vec<f64> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    if cols != vector.len() {
        panic!("Cannot perform dot product: incompatible dimensions");
    }

    let mut result = vec![0.0; rows];

    for i in 0..rows {
        for j in 0..cols {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    result
}

pub fn dot_product_single(matrix: &Vec<f64>, vector: &Vec<f64>) -> f64 {
    let cols = matrix.len();

    if cols != vector.len() {
        panic!("Cannot perform dot product: incompatible dimensions");
    }

    let mut result = 0.0;

    for i in 0..cols {
        result += matrix[i] * vector[i];
    }

    result
}

pub fn argmax(matrix: &Vec<f64>) -> usize {
    let mut curr_max = std::f64::MIN;
    let mut curr_index = 0;

    for i in 0..matrix.len() {
        if curr_max <= matrix[i] {
            curr_max = matrix[i];
            curr_index = i;
        }
    }
    curr_index
}