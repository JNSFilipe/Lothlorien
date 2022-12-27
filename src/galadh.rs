use crate::heart_wood::{HeartWood, sow_tree};

#[derive(Debug)]
pub struct Galadh {
    root: Option<HeartWood>
}


impl Galadh {
    pub fn new() -> Galadh {
        Galadh {
            root: None
        }
    }

    pub fn sow(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>) {
        self.root = sow_tree(x ,y);
    }

    pub fn print(&self) {
        let root = self.root.as_ref().unwrap();
        root.print(0);
    }

    pub fn predict_single(&self, x: &Vec<f64>) -> Option<usize> {
        let root = self.root.as_ref().unwrap();
        root.predict_single(x)
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        let root = self.root.as_ref().unwrap();
        root.predict(x)
    }

    pub fn get_root_threshold(&self) -> Option<f64> {
        let root = self.root.as_ref().unwrap();
        root.get_threshold()
    }

    pub fn get_root_feature(&self) -> Option<Vec<f64>> {
        let root = self.root.as_ref().unwrap();
        root.get_feature()
    }
}