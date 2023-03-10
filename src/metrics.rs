pub fn gini_impurity(y: &[usize]) -> f64 {
    let mut classes = [0, 0];
    for &label in y {
        classes[label] += 1;
    }

    let sum: f64 = classes.iter().map(|&x| x as f64).sum();
    1.0 - classes.iter().map(|&x| (x as f64 / sum).powi(2)).sum::<f64>()
}

pub fn accuracy(y: &[usize], y_hat: &[usize]) -> f64 {
    let mut aux = Vec::new();
    for i in 0..y.len() {
        let l = if y[i]==y_hat[i] { 1 } else { 0 };
        aux.push(l);
    }

    (aux.iter().sum::<i32>() as f64 / aux.len() as f64) * 100.0
}

pub fn mse(y: &Vec<f64>, y_hat: &Vec<f64>) -> f64 {
    y.iter()
        .zip(y_hat.iter())
        .map(|(y, y_hat)| ((*y as f64) - *y_hat).powf(2.0))
        .sum::<f64>()
        / y.len() as f64
}