#include <TFile.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <fstream>
#include <sstream>
#include <string>

void plot_hists() {
    // ---------------------------------------------------------------
    // 1. Load coincedence theta histogram (theta_hit_bins.txt)
    // ---------------------------------------------------------------
    TH1D *h_theta = new TH1D("h_theta", "Theta Distribution (Hits);Theta [deg];Counts", 
                            72, -90, 90); //Set Bins
    Double_t theta_total = 0;

    std::ifstream theta_file("theta_hit_bins.txt");
    std::string theta_line;
    int theta_bin = 1;

    while (std::getline(theta_file, theta_line)) {
        if (theta_line[0] == '#') continue;
        float bin_center;
        int count;
        std::istringstream theta_iss(theta_line);
        if (theta_iss >> bin_center >> count) {
            h_theta->SetBinContent(theta_bin, count);
            theta_total += count;  // Accumulate total counts
            theta_bin++;
        }
    }
    theta_file.close();
    h_theta->SetEntries(theta_total);  // Set entries = sum of bin contents

    // ---------------------------------------------------------------
    // 2. Load coincedence phi histogram (phi_hit_bins.txt)
    // ---------------------------------------------------------------
    TH1D *h_phi = new TH1D("h_phi", "Phi Distribution (Hits);Phi [deg];Counts", 
                          360, 0, 360); //Set Bins
    Double_t phi_total = 0;

    std::ifstream phi_file("phi_hit_bins.txt");
    std::string phi_line;
    int phi_bin = 1;

    while (std::getline(phi_file, phi_line)) {
        if (phi_line[0] == '#') continue;
        float bin_center;
        int count;
        std::istringstream phi_iss(phi_line);
        if (phi_iss >> bin_center >> count) {
            h_phi->SetBinContent(phi_bin, count);
            phi_total += count;  // Accumulate total counts
            phi_bin++;
        }
    }
    phi_file.close();
    h_phi->SetEntries(phi_total);  // Set entries = sum of bin contents

    // ---------------------------------------------------------------
    // 3. Draw and verify
    // ---------------------------------------------------------------
    TCanvas *c1 = new TCanvas("c1", "Histograms", 1200, 600);
    c1->Divide(2,1);

    c1->cd(1);
    h_theta->Draw();
    std::cout << "θ entries: " << h_theta->GetEntries() << std::endl;

    c1->cd(2);
    h_phi->Draw();
    std::cout << "φ entries: " << h_phi->GetEntries() << std::endl;

    // Save to file
    TFile out_file("coincedence.root", "RECREATE");
    h_theta->Write();
    h_phi->Write();
    out_file.Close();
}
