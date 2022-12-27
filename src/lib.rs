mod metrics;
mod algebra;
mod heart_wood;
mod galadh;

use rand::Rng;

use crate::metrics::accuracy;
use crate::algebra::argmax;
use crate::galadh::Galadh;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_simple_1d_threshold_cut() {
        let fixed_feature = 1;
        let fixed_trheshold = 15.0;

        let mut rng = rand::thread_rng();

        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..1000 {
            let sample = vec![rng.gen_range(0.0..50.0), rng.gen_range(0.0..50.0)];
            let label = if sample[fixed_feature] < fixed_trheshold { 1 } else { 0 };
            x.push(sample);
            y.push(label);
        }

        let mut tree = Galadh::new();
        tree.sow(&x, &y);
        
        assert_eq!(argmax(&(tree.get_root_feature().unwrap())), fixed_feature);
        assert!((tree.get_root_threshold().unwrap() - fixed_trheshold).abs() < fixed_trheshold*0.15);
    }

    #[test]
    fn accuracy_with_2d_threshold() {
        let mut rng = rand::thread_rng();

        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..1000 {
            let sample = vec![rng.gen_range(0.0..50.0), rng.gen_range(0.0..50.0)];
            let label = if sample[0] + sample[1] < 15.0 { 0 } else { 1 };
            x.push(sample);
            y.push(label);
        }

        let mut tree = Galadh::new();
        tree.sow(&x, &y);

        let y_hat = tree.predict(&x);

        assert_eq!(accuracy(&y.as_slice(), &y_hat.as_slice()), 100.0);
    }

    #[test]
    fn print_test() {
        let mut rng = rand::thread_rng();

        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..1000 {
            let sample = vec![rng.gen_range(0.0..50.0), rng.gen_range(0.0..50.0)];
            let label = if sample[0] + sample[1] < 15.0 { 0 } else { 1 };
            x.push(sample);
            y.push(label);
        }

        let mut tree = Galadh::new();
        tree.sow(&x, &y);
        tree.print();

        assert!(true);
    }
}
