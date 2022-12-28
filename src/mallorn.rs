use rayon::prelude::*;
use crate::galadh::Galadh;

pub struct Mallorn {
    trees: Vec<Galadh>,
    learning_rate: f64
}

impl Mallorn {
    pub fn new(n_trees: usize, learning_rate: f64) -> Mallorn {
        let trees = (0..n_trees).map(|_| Galadh::new()).collect();
        Mallorn { trees, learning_rate }
    }

    pub fn sow(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>) {
        // TODO: in order for this to work properly, probably y must be changed to Vec<f64> in Galadh...
        let mut residuals: Vec<f64> = y.to_vec().iter().map(|x| *x as f64).collect();

        for tree in &mut self.trees {
            let predictions = tree.predict(x);

            // Compute the residuals
            for i in 0..residuals.len() {
                residuals[i] -= self.learning_rate * (predictions[i] as f64);
            }

            // Fit the tree on the residuals
            let r = residuals.iter().map(|x| *x as usize).collect();
            tree.sow(x, &r);
        }
    }

    pub fn predict_single(&self, x: &Vec<f64>) -> Option<usize> {
        let mut prediction = 0.0;

        for tree in &self.trees { 
            if let Some(p) = tree.predict_single(x) {
                prediction += self.learning_rate * p as f64;
            }
        }

        if prediction > 0.5 {
            Some(1)
        } else {
            Some(0)
        }
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<usize> {
        x.par_iter().map(|x| self.predict_single(x).unwrap()).collect()
    }
}