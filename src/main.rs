extern crate plotters;
use plotters::prelude::*;
use qcorrelations as qcorr;

fn plot_wernerstates_discord() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/wernerstates_gdiscord.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE);
    let root = root.margin(1, 0, 0, 1);
    let mut chart = ChartBuilder::on(&root)
        .caption("Geometric (Red) and Trace Distance (Blue) Discord for Werner States as a function of λ", ("sans-serif", 20).into_font())
        .x_label_area_size(20)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..1f32, 0f32..1f32)?;

    chart
        .configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .y_label_formatter(&|x| format!("{:.3}", x))
        .draw()?;

    let mut discords: Vec<(f32,f32)> = Vec::new();
    let mut lmda = 0.0;
    println!("Geometric Discord by Random Search:");
    while lmda <= 1.01 {
        println!("Computing for λ = {}", &lmda);
        let w_l = qcorr::werner_state(lmda);
        let discord = qcorr::geometric_discord(&w_l, 1000000) as f32;
        discords.push((lmda as f32, discord));
        println!("{}", discord);
        lmda += 0.05;
    }
    chart.draw_series(LineSeries::new(
        discords,
        &RED,
    ))?;
    let mut tdds: Vec<(f32,f32)> = Vec::new();
    lmda = 0.0;
    println!("Trace Distance Discord by Random Search:");
    while lmda <= 1.01 {
        println!("Computing for λ = {}", &lmda);
        let w_l = qcorr::werner_state(lmda);
        let discord = qcorr::trace_distance_discord(&w_l, 20000000) as f32;
        tdds.push((lmda as f32, discord));
        println!("{}", discord);
        lmda += 0.1;
    }
    chart.draw_series(LineSeries::new(
        tdds,
        &BLUE,
    ))?;
    Ok(())
}

fn main() {
    plot_wernerstates_discord();
}
