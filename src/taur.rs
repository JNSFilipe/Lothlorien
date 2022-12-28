use rayon::prelude::*;
use crate::galadh::Galadh;
use rand::{Rng, thread_rng, seq::SliceRandom};

pub struct Taur {
    trees: Vec<Galadh>
}

impl Taur {
    pub fn new(n_trees: usize) -> Taur {
        let trees = (0..n_trees).map(|_| Galadh::new()).collect();
        Taur { trees }
    }

    pub fn sow(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>, bootstrapping: bool) {
        // TODO: create a way to define the random seed
        let n_samples = x.len();
        let sample_size = (n_samples as f64 * 0.6) as usize;

        self.trees.par_iter_mut().for_each(|tree| {
            let mut rng = rand::thread_rng();
            let mut x_sample = Vec::new();
            let mut y_sample = Vec::new();

            if bootstrapping {
                // Use bootstrapping to sample with replacement
                for _ in 0..sample_size {
                    let index = rng.gen_range(0..n_samples);
                    x_sample.push(x[index].clone());
                    y_sample.push(y[index]);
                }
            } else {
                // Sample without replacement
                // TODO: The way things are working now, training without bootstrapping results in pretty much training the same trees n_trees times.
                // in order to create trully extremely randomized trees, the sow method of Galadh needs to be changed to do it the ERT way
                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(&mut rng);
                let indices = &indices[..sample_size];
                for index in indices {
                    x_sample.push(x[*index].clone());
                    y_sample.push(y[*index]);
                }
            }

            tree.sow(&x_sample, &y_sample);
        });
    }

    pub fn predict_single(&self, x: &Vec<f64>) -> Option<usize> {
        let predictions: Vec<Option<usize>> = self.trees.par_iter().map(|tree| tree.predict_single(x)).collect();

        let mut counts = [0; 2];
        for prediction in predictions {
            if let Some(p) = prediction {
                counts[p] += 1;
            }
        }

        if counts[0] > counts[1] {
            Some(0)
        } else {
            Some(1)
        }
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<usize> {
        x.par_iter().map(|x| self.predict_single(x).unwrap()).collect()
    }
}
