

void DrawNNResults(std::string filename_train,std::string filename_test,double thr=0.5)
{
	double nobjects;
	double type;
	double rectype;
	double x;
	double y;
	double xrec;
	double yrec;

	//Read train data
	TTree* data_train= new TTree("data_train","data_train");
	data_train->ReadFile(filename_train.c_str(),"nobjects/D:type/D:rectype/D:x/D:xrec/D:y/D:yrec/D");
	//data_train->ReadFile(filename_train.c_str(),"type/D:rectype/D:x/D:xrec/D:y/D:yrec/D");

	//data_train->SetBranchAddress("nobjects",&nobjects);
	data_train->SetBranchAddress("type",&type);
	data_train->SetBranchAddress("rectype",&rectype);
	data_train->SetBranchAddress("x",&x);
	data_train->SetBranchAddress("y",&y);
	data_train->SetBranchAddress("xrec",&xrec);
	data_train->SetBranchAddress("yrec",&yrec);
	
	//Read test data
	TTree* data_test= new TTree("data_test","data_test");
	data_test->ReadFile(filename_test.c_str(),"nobjects/D:type/D:rectype/D:x/D:xrec/D:y/D:yrec/D");
	//data_test->ReadFile(filename_test.c_str(),"type/D:rectype/D:x/D:xrec/D:y/D:yrec/D");
		
	//data_test->SetBranchAddress("nobjects",&nobjects);
	data_test->SetBranchAddress("type",&type);
	data_test->SetBranchAddress("rectype",&rectype);
	data_test->SetBranchAddress("x",&x);
	data_test->SetBranchAddress("y",&y);
	data_test->SetBranchAddress("xrec",&xrec);
	data_test->SetBranchAddress("yrec",&yrec);

	//Fill histograms
	TH1D* bkgHisto_train= new TH1D("bkgHisto_train","",100,-0.5,1.5);
	bkgHisto_train->Sumw2();

	TH1D* sourceHisto_train= new TH1D("sourceHisto_train","",100,-0.5,1.5);
	sourceHisto_train->Sumw2();

	TH1D* bkgHisto_test= new TH1D("bkgHisto_test","",100,-0.5,1.5);
	bkgHisto_test->Sumw2();

	TH1D* sourceHisto_test= new TH1D("sourceHisto_test","",100,-0.5,1.5);
	sourceHisto_test->Sumw2();

	TH2D* sourcePosPull_train= new TH2D("sourcePosPull_train","",100,-20,20,100,-20,20);
	sourcePosPull_train->Sumw2();

	TH2D* sourcePosPull_test= new TH2D("sourcePosPull_test","",100,-20,20,100,-20,20);
	sourcePosPull_test->Sumw2();

	//- Train data
	double nSources_tot= 0;
	double nSources_det= 0;
	double nSources_rec= 0;
	double nSources_true= 0;
	double nSources_false= 0;

	for(int i=0;i<data_train->GetEntries();i++)
	{
		data_train->GetEntry(i);

		//Compute completeness
		cout<<"type="<<type<<", rectype="<<rectype<<", x="<<x<<", xrec="<<xrec<<", dx="<<xrec-x<<", y="<<y<<", yrec="<<yrec<<", dy="<<yrec-y<<endl;
		if(type==0){
			bkgHisto_train->Fill(rectype);
		}
		else if(type==1){	
			sourceHisto_train->Fill(rectype);
			nSources_tot++;
			if(rectype>thr) nSources_det++;
		}

		//Compute reliability
		if(rectype>thr){
			nSources_rec++;
			if(type==0) nSources_false++;
			else if(type==1) {
				nSources_true++;
				sourcePosPull_train->Fill(xrec-x,yrec-y);
			}
		}

		

	}//end loop entries

	double completeness_train= 0;
	if(nSources_tot>0) completeness_train= nSources_det/nSources_tot;
	double reliability_train= 0;
	if(nSources_rec>0) reliability_train= nSources_true/nSources_rec;
	
	cout<<"INFO: Train completeness="<<completeness_train<<", reliability="<<reliability_train<<endl;

	//- Test data
	nSources_tot= 0;
	nSources_det= 0;
	nSources_rec= 0;
	nSources_true= 0;
	nSources_false= 0;

	for(int i=0;i<data_test->GetEntries();i++)
	{
		data_test->GetEntry(i);

		//Compute completeness
		if(type==0){
			bkgHisto_test->Fill(rectype);
		}
		else if(type==1){
			sourceHisto_test->Fill(rectype);
			nSources_tot++;
			if(rectype>thr) nSources_det++;
		}

		//Compute reliability
		if(rectype>thr){
			nSources_rec++;
			if(type==0) nSources_false++;
			else if(type==1) {
				nSources_true++;
				sourcePosPull_test->Fill(xrec-x,yrec-y);
			}
		}


	}//end loop entries

	double completeness_test= 0;
	if(nSources_tot>0) completeness_test= nSources_det/nSources_tot;
	double reliability_test= 0;
	if(nSources_rec>0) reliability_test= nSources_true/nSources_rec;
	
	cout<<"INFO: Test completeness="<<completeness_test<<", reliability="<<reliability_test<<endl;

	//Draw plots
	gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadBottomMargin(0.12);
  gStyle->SetPadLeftMargin(0.12);
  gStyle->SetPadRightMargin(0.12);

	TCanvas* Plot= new TCanvas("Plot","Plot",1200,1600);
	Plot->cd();
	Plot->Divide(2,2);
	
	//- Train data
	Plot->cd(1);

	TH2D* PlotBkg_1= new TH2D("PlotBkg_1","",100,-0.5,1.5,100,0,1.1);
	PlotBkg_1->SetStats(0);	
	PlotBkg_1->SetXTitle("Sourceness");
	PlotBkg_1->SetYTitle("entries");
	PlotBkg_1->Draw();

	bkgHisto_train->SetStats(0);
	bkgHisto_train->SetLineColor(kBlack);	
	bkgHisto_train->SetFillColor(kBlack);
	bkgHisto_train->SetFillStyle(3004);
	bkgHisto_train->SetLineWidth(1);
	bkgHisto_train->DrawNormalized("hist same");

	sourceHisto_train->SetStats(0);
	sourceHisto_train->SetLineColor(kRed);
	sourceHisto_train->SetFillColor(kRed);
	sourceHisto_train->SetFillStyle(3008);
	sourceHisto_train->SetLineWidth(1);
	sourceHisto_train->DrawNormalized("hist same");

	TLegend* PlotLegend_1= new TLegend(0.6,0.7,0.7,0.8);
	PlotLegend_1->SetFillColor(0);
	PlotLegend_1->SetTextSize(0.045);
	PlotLegend_1->SetTextFont(52);
	PlotLegend_1->SetHeader("train");
	PlotLegend_1->AddEntry(bkgHisto_train,"bkg","l");
	PlotLegend_1->AddEntry(sourceHisto_train,"sources","l");
	PlotLegend_1->Draw("same");

	//- Test data
	Plot->cd(2);

	TH2D* PlotBkg_2= new TH2D("PlotBkg_2","",100,-0.5,1.5,100,0,1.1);
	PlotBkg_2->SetStats(0);	
	PlotBkg_2->SetXTitle("Sourceness");
	PlotBkg_2->SetYTitle("entries");
	PlotBkg_2->Draw();

	bkgHisto_test->SetStats(0);
	bkgHisto_test->SetLineColor(kBlack);
	bkgHisto_test->SetFillColor(kBlack);
	bkgHisto_test->SetFillStyle(3004);
	bkgHisto_test->SetLineWidth(1);
	bkgHisto_test->DrawNormalized("hist same");

	sourceHisto_test->SetStats(0);
	sourceHisto_test->SetLineColor(kRed);
	sourceHisto_test->SetFillColor(kRed);
	sourceHisto_test->SetFillStyle(3008);
	sourceHisto_test->SetLineWidth(1);
	sourceHisto_test->DrawNormalized("hist same");

	TLegend* PlotLegend_2= new TLegend(0.6,0.7,0.7,0.8);
	PlotLegend_2->SetFillColor(0);
	PlotLegend_2->SetTextSize(0.045);
	PlotLegend_2->SetTextFont(52);
	PlotLegend_2->SetHeader("test");
	PlotLegend_2->AddEntry(bkgHisto_test,"bkg","l");
	PlotLegend_2->AddEntry(sourceHisto_test,"sources","l");
	PlotLegend_2->Draw("same");

	//- Train data
	Plot->cd(3);
	
	sourcePosPull_train->SetXTitle("#deltax");
	sourcePosPull_train->SetYTitle("#deltay");
	sourcePosPull_train->Draw("COLZ");

	//- Test data
	Plot->cd(4);
	
	sourcePosPull_test->SetXTitle("#deltax");
	sourcePosPull_test->SetYTitle("#deltay");
	sourcePosPull_test->Draw("COLZ");
	

}//close macro


