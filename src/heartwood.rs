struct BackProp {
    lr: f64,
    W: Vec<f64>,
    k: f64,
}

impl BackProp {
    fn new(lr: f64) -> BackProp {
        BackProp { lr, W: Vec::new(), k: 0.0 }
    }

    fn next_batch<'a>(&'a self, x: &'a Vec<Vec<f64>>, y: &'a Vec<usize>, batch_size: usize) -> impl Iterator<Item=(&'a Vec<Vec<f64>>, &'a Vec<usize>)> {
        (0..x.len()).step_by(batch_size).map(move |i| (&x[i..i+batch_size], &y[i..i+batch_size]))
    }

    fn lr(&self, iter: usize, total_iter: usize, fixed: bool) -> f64 {
        if fixed {
            self.lr
        } else {
            (self.lr * total_iter as f64) / (iter + total_iter) as f64
        }
    }

    fn sigmoid(x: &Vec<f64>, derivative: bool) -> Vec<f64> {
        if derivative {
            x.iter().map(|x| (-x).exp() / (1.0 + (-x).exp()).powi(2)).collect()
        } else {
            x.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
        }
    }

    fn tanh(x: &Vec<f64>, derivative: bool) -> Vec<f64> {
        if derivative {
            x.iter().map(|x| 0.5 / x.cosh().powi(2)).collect()
        } else {
            x.iter().map(|x| (x.tanh() + 1.0) / 2.0).collect()
        }
    }

    fn mse(y: &Vec<usize>, y_hat: &Vec<f64>) -> f64 {
        y.iter().zip(y_hat).map(|(y, y_hat)| (y - y_hat).powi(2)).sum::<f64>() / y.len() as f64
    }

    fn backprop(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>, lr: Option<f64>) -> f64 {
        let lr = lr.unwrap_or(self.lr);

        let y_hat = self.f(x);
        let y: Vec<f64> = y.iter().map(|y| *y as f64).collect();
        let c = Self::mse(&y, &y_hat);
        let e: Vec<f64> = y.iter().zip(y_hat).map(|(y, y_hat)| y - y_hat).collect();

        let d_w: Vec<f64> = self.f(x, derivative: Some("W"))
            .iter()
            .zip(e.iter())
            .map(|(f, e)| -2.0 * f * e)
            .sum::<f64>() / y.len() as f64;
        let d_b: f64 = self.f(x, derivative: Some("b"))
            .iter()
            .zip(e.iter())
            .map(|(f, e)| -2.0 * f * e)
            .sum::<f64>() / y.len() as f64;

        self.W = self.W.iter().zip(d_w.iter()).map(|(w, d_w)| w - lr * d_w).collect();
        self.k -= lr * d_b;

        c
    }

    fn train(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>, batch_size: usize, epochs: usize) {
        let mut epoch_loss = Vec::new();
        let mut epoch_accuracy = Vec::new();

        for epoch in 0..epochs {
            let mut batch_loss = Vec::new();
            let mut batch_accuracy = Vec::new();

            let lr = self.lr(epoch, epochs, fixed: true);

            for (x_batch, y_batch) in self.next_batch(x, y, batch_size) {
                let loss = self.backprop(x_batch, y_batch, lr: Some(lr));
                batch_loss.push(loss);
                batch_accuracy.push(self.accuracy(x_batch, y_batch));
            }

            epoch_loss.push(batch_loss.iter().sum::<f64>() / batch_loss.len() as f64);
            epoch_accuracy.push(batch_accuracy.iter().sum::<f64>() / batch_accuracy.len() as f64);
            println!(
                "Epoch {}: Loss: {:.6} Accuracy: {:.2}%",
                epoch + 1,
                epoch_loss[epoch],
                epoch_accuracy[epoch] * 100.0
            );
        }
    }

    fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>, batch_size: usize, epochs: usize, metric: &str) {
        let num_feat = x[0].len();
        self.W = (0..num_feat)
            .map(|_| thread_rng().gen_range(-1.0, 1.0))
            .collect();
        self.k = thread_rng().gen_range(-1.0, 1.0);

        let parent_imp = self.info_metrics(y, metric);
        self.train(x, y, batch_size, epochs);
    }

    fn accuracy(&self, x: &Vec<Vec<f64>>, y: &Vec<usize>) -> f64 {
        let y_hat = self.predict(x);
        y.iter().zip(y_hat.iter()).filter(|(y, y_hat)| y == y_hat).count() as f64 / y.len() as f64
    }

    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<usize> {
        self.f(x).iter().map(|y| y.round() as usize).collect()
    }

    fn f(&self, x: &Vec<Vec<f64>>, derivative: Option<&str>) -> Vec<f64> {
        let d = derivative.unwrap_or("");
        if d == "W" {
            x.iter().map(|x| x.iter().zip(self.W.iter()).map(|(x, w)| x * w).sum()).collect()
        } else if d == "b" {
            x.iter().map(|_| self.k).collect()
        } else {
            x.iter().map(|x| x.iter().zip(self.W.iter()).map(|(x, w)| x * w).sum() + self.k).collect()
        }
    }
}


