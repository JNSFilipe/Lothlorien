use crate::algebra::{dot_product, dot_product_single};
use crate::metrics::accuracy;

struct BackProp {
    lr: f64,
    w: Vec<f64>,
    k: f64,
}

impl BackProp {
    fn next_batch(&self, x: &Vec<Vec<f64>>, y: &Vec<usize>, batch_size: usize) -> Vec<(Vec<Vec<f64>>, Vec<usize>)> {
        let mut batches = Vec::new();
        for i in 0..x.len() / batch_size {
            let start = i * batch_size;
            let end = (i + 1) * batch_size;
            batches.push((x[start..end].to_vec(), y[start..end].to_vec()));
        }
        batches
    }

    pub fn forward_single(&self, x: &Vec<f64>, derivative: Option<String>) -> bool {
        if derivative.unwrap() == "f" {
            self.tanh(dot_product_single(x, &self.w) - self.k, false)
        } else if derivative.unwrap() == "w" {
            x * self.tanh(dot_product_single(x, &self.w) - self.k, true)
        } else if derivative.unwrap() == "b" {
            -1 * self.tanh(dot_product_single(x, &self.w) - self.k, true)
        } else {
            dot_product_single(x, &self.w) < self.k
        }
    }

    pub fn forward(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        let mut results = Vec::new();
        for s in x.iter(){
            results.push(self.forward_single(s) as i32 as f64);
        }
        results
    }

    fn lr(&self, iter: usize, total_iter: usize, fixed: bool) -> f64 {
        if fixed {
            self.lr
        } else {
            (self.lr * total_iter as f64) / ((iter + total_iter) as f64)
        }
    }

    fn tanh(&self, x: &Vec<f64>, derivative: bool) -> Vec<f64> {
        if derivative {
            x.iter()
                .map(|x| 0.5 / (x.cosh().powf(2.0)))
                .collect()
        } else {
            x.iter().map(|x| (x.tanh() + 1.0) / 2.0).collect()
        }
    }

    fn backprop(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>, lr: Option<f64>) -> f64 {
        let lr = lr.unwrap_or(self.lr);
        let y_hat = self.forward(x);
        let y: Vec<f64> = y.iter().map(|y| *y as f64).collect();
        let c = mse(&y, &y_hat);
        let e = y.iter().zip(y_hat.iter()).map(|(y, y_hat)| y - y_hat).collect::<Vec<f64>>();

        let d_w = -2.0
            * x
                .iter()
                .zip(e.iter())
                .map(|(x, e)| x.iter().zip(self.w.iter()).map(|(x, w)| x * w).sum() * e)
                .sum::<f64>()
            / y.len() as f64;
        let d_b = -2.0
            * e
                .iter()
                .map(|e| self.k)
                .sum::<f64>()
            / y.len() as f64;

        self.w = self.w.iter().zip(self.forward(x, Some("W")).iter()).map(|(w, d_w)| w - lr * d_w).collect();
        self.k -= lr * d_b;

        c
    }

    fn train(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>, batch_size: usize, epochs: usize) {
        let mut epoch_loss = Vec::new();
        let mut epoch_accuracy = Vec::new();

        for epoch in 0..epochs {
            let mut batch_loss = Vec::new();
            let mut batch_accuracy = Vec::new();

            let lr = self.lr(epoch, epochs, true);

            for (x_batch, y_batch) in self.next_batch(x, y, batch_size) {
                let loss = self.backprop(&x_batch, &y_batch, Some(lr));
                batch_loss.push(loss);
                batch_accuracy.push(accuracy(&x_batch, &y_batch));
            }

            epoch_loss.push(batch_loss.iter().sum::<f64>() / batch_loss.len() as f64);
            epoch_accuracy.push(
                batch_accuracy.iter().sum::<f64>() / batch_accuracy.len() as f64,
            );
        }
    }
}
