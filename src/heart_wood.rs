use crate::algebra::dot_product_single;
use crate::metrics::gini_impurity;

#[derive(Debug)]
pub struct HeartWood {
    feature: Vec<f64>,
    threshold: f64,
    left: Option<Box<HeartWood>>,
    right: Option<Box<HeartWood>>,
    label: Option<usize>,
}

impl HeartWood {
    fn new(feature: Vec<f64>, threshold: f64) -> HeartWood {
        HeartWood {
            feature,
            threshold,
            left: None,
            right: None,
            label: None,
        }
    }

    fn set_left(&mut self, left: HeartWood) {
        self.left = Some(Box::new(left));
    }

    fn set_right(&mut self, right: HeartWood) {
        self.right = Some(Box::new(right));
    }

    fn set_label(&mut self, label: usize) {
        self.label = Some(label);
    }

    fn judge(&self, x: &Vec<f64>) -> bool {
        dot_product_single(x, &self.feature) < self.threshold
    }

    pub fn get_threshold(&self) -> Option<f64> {
        Some(self.threshold)
    }

    pub fn get_feature(&self) -> Option<Vec<f64>> {
        Some(self.feature.clone())
    }

    pub fn predict_single(&self, x: &Vec<f64>) -> Option<usize> {
        if let Some(label) = self.label {
            return Some(label);
        }

        //let feature_value = x[self.feature];
        //if feature_value < self.threshold {
        if self.judge(x) {
            if let Some(ref left) = self.left {
                return left.predict_single(x);
            }
        } else {
            if let Some(ref right) = self.right {
                return right.predict_single(x);
            }
        }

        None
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        let mut results = Vec::new();
        for s in x.iter(){
            results.push(self.predict_single(s).unwrap());
        }
        results
    }

    pub fn print(&self, depth: usize) {
        let mut indent = String::new();
        for _ in 0..depth {
            indent.push_str("  ");
        }

        if let Some(label) = self.label {
            println!("{}[{}]", indent, label);
            return;
        }

        println!("{}[X{:?} < {:.2}]", indent, self.feature, self.threshold);
        if let Some(ref left) = self.left {
            println!("{}[left]", indent);
            left.print(depth + 1);
        }
        if let Some(ref right) = self.right {
            println!("{}[right]", indent);
            right.print(depth + 1);
        }
    }
}

fn best_split(x: &Vec<Vec<f64>>, y: &Vec<usize>) -> Option<(Vec<f64>, f64)> {
    let n_samples = x.len();
    let n_features = x[0].len();

    let mut best_feature = vec![0.0; n_features];
    let mut best_threshold = 0.0;
    let mut best_impurity = 1.0;

    let mut weights = vec![0.0; n_features];

    for feature in 0..n_features {

        // Reset weights to all zeros
        for i in 0..n_features {
            weights[i] = 0.0;
        }
        
        weights[feature] = 1.0;

        let mut thresholds = Vec::new();
        for sample in x {
            thresholds.push(sample[feature]);
        }
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        thresholds.dedup();

        for threshold in thresholds {
            let mut left_y = Vec::new();
            let mut right_y = Vec::new();

            for i in 0..n_samples {
                if dot_product_single(&x[..][i], &weights) < threshold {
                    left_y.push(y[i]);
                } else {
                    right_y.push(y[i]);
                }
            }

            let impurity =
                (left_y.len() as f64 / n_samples as f64) * gini_impurity(&left_y)
                    + (right_y.len() as f64 / n_samples as f64) * gini_impurity(&right_y);

            if impurity < best_impurity {
                best_feature = weights.clone();
                best_threshold = threshold;
                best_impurity = impurity;
            }
        }
    }

    if best_impurity < 1.0 {
        Some((best_feature, best_threshold))
    } else {
        None
    }
}

pub fn sow_tree(x: &Vec<Vec<f64>>, y: &Vec<usize>) -> Option<HeartWood> {
    let n_samples = x.len();
    let n_features = x[0].len();
    let mut classes = [0, 0];
    for &label in y {
        classes[label] += 1;
    }

    if classes[0] == n_samples {
        let mut aux = vec![0.0; n_features];
        aux[0] = 1.0;
        let mut HeartWood = HeartWood::new(aux, 0.0);
        HeartWood.set_label(0);
        return Some(HeartWood);
    }

    if classes[1] == n_samples {
        let mut aux = vec![0.0; n_features];
        aux[0] = 1.0;
        let mut HeartWood = HeartWood::new(aux, 0.0);
        HeartWood.set_label(1);
        return Some(HeartWood);
    }

    if let Some((weights, threshold)) = best_split(x, y) {
        let mut left_x = Vec::new();
        let mut left_y = Vec::new();
        let mut right_x = Vec::new();
        let mut right_y = Vec::new();

        for i in 0..n_samples {
            let v = dot_product_single(&x[..][i], &weights);
            if dot_product_single(&x[..][i], &weights) < threshold {
                left_x.push(x[i].clone());
                left_y.push(y[i]);
            } else {
                right_x.push(x[i].clone());
                right_y.push(y[i]);
            }
        }

        let mut HeartWood = HeartWood::new(weights, threshold);
        if let Some(left) = sow_tree(&left_x, &left_y) {
            HeartWood.set_left(left);
        }
        if let Some(right) = sow_tree(&right_x, &right_y) {
            HeartWood.set_right(right);
        }
        return Some(HeartWood);
    }

    None
}