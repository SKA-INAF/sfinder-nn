

void DrawNNLossVSEpochs(std::string filename)
{
	//Read data
	TTree* data= new TTree("data","data");
	data->ReadFile(filename.c_str(),"epoch/D:loss_train_type/D:loss_train_pars/D:loss_train/D:loss_test_type/D:loss_test_pars/D:loss_test/D:acc_train/D:acc_test/D");

	double epoch;
	double loss_train_type;
	double loss_train_pars;
	double loss_train;
	double loss_test_type;
	double loss_test_pars;
	double loss_test;
	double acc_train;
	double acc_test;

	data->SetBranchAddress("epoch",&epoch);
	data->SetBranchAddress("loss_train_type",&loss_train_type);
	data->SetBranchAddress("loss_train_pars",&loss_train_pars);
	data->SetBranchAddress("loss_train",&loss_train);
	data->SetBranchAddress("loss_test_type",&loss_test_type);
	data->SetBranchAddress("loss_test_pars",&loss_test_pars);
	data->SetBranchAddress("loss_test",&loss_test);
	
	//Fill graphs
	TGraph* lossVSEpoch_train= new TGraph;
	lossVSEpoch_train->SetLineColor(kRed);
	lossVSEpoch_train->SetLineWidth(2);

	TGraph* lossVSEpoch_test= new TGraph;
	lossVSEpoch_test->SetLineColor(kGreen+1);
	lossVSEpoch_test->SetLineWidth(2);

	TGraph* lossTypeVSEpoch_train= new TGraph;
	lossTypeVSEpoch_train->SetLineColor(kRed);
	lossTypeVSEpoch_train->SetLineWidth(2);

	TGraph* lossTypeVSEpoch_test= new TGraph;
	lossTypeVSEpoch_test->SetLineColor(kGreen+1);
	lossTypeVSEpoch_test->SetLineWidth(2);

	TGraph* lossParsVSEpoch_train= new TGraph;
	lossParsVSEpoch_train->SetLineColor(kRed);
	lossParsVSEpoch_train->SetLineWidth(2);

	TGraph* lossParsVSEpoch_test= new TGraph;
	lossParsVSEpoch_test->SetLineColor(kGreen+1);
	lossParsVSEpoch_test->SetLineWidth(2);

	double loss_min= 1.e+99;
	double loss_max= -1.e+99;
	double loss_type_min= 1.e+99;
	double loss_type_max= -1.e+99;
	double loss_pars_min= 1.e+99;
	double loss_pars_max= -1.e+99;

	for(int i=0;i<data->GetEntries();i++)
	{
		data->GetEntry(i);
		if(loss_train<loss_min) loss_min= loss_train;
		if(loss_test<loss_min) loss_min= loss_test;
		if(loss_train>loss_max) loss_max= loss_train;
		if(loss_test>loss_max) loss_max= loss_test;

		if(loss_train_type<loss_type_min) loss_type_min= loss_train_type;
		if(loss_test_type<loss_type_min) loss_type_min= loss_test_type;
		if(loss_train_type>loss_type_max) loss_type_max= loss_train_type;
		if(loss_test_type>loss_type_max) loss_type_max= loss_test_type;

		if(loss_train_pars<loss_pars_min) loss_pars_min= loss_train_pars;
		if(loss_test_pars<loss_pars_min) loss_pars_min= loss_test_pars;
		if(loss_train_pars>loss_pars_max) loss_pars_max= loss_train_pars;
		if(loss_test_pars>loss_pars_max) loss_pars_max= loss_test_pars;

		lossVSEpoch_train->SetPoint(i,epoch,loss_train);
		lossVSEpoch_test->SetPoint(i,epoch,loss_test);
		lossTypeVSEpoch_train->SetPoint(i,epoch,loss_train_type);
		lossTypeVSEpoch_test->SetPoint(i,epoch,loss_test_type);
		lossParsVSEpoch_train->SetPoint(i,epoch,loss_train_pars);
		lossParsVSEpoch_test->SetPoint(i,epoch,loss_test_pars);		

	}//end loop epochs

	double dloss= 0.1*fabs(loss_max-loss_min);
	double dloss_type= 0.1*fabs(loss_type_max-loss_type_min);
	double dloss_pars= 0.1*fabs(loss_pars_max-loss_pars_min);

	//Draw graphs
	gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadBottomMargin(0.12);
  gStyle->SetPadLeftMargin(0.12);
  gStyle->SetPadRightMargin(0.12);

	TCanvas* Plot= new TCanvas("Plot","Plot",1200,600);
	Plot->cd();
	Plot->Divide(3,1);
	
	//- Total loss
	Plot->cd(1);

	TH2D* PlotBkg_1= new TH2D("PlotBkg_1","",100,0,data->GetEntries()+1,100,loss_min-dloss,loss_max+dloss);
	PlotBkg_1->SetStats(0);	
	PlotBkg_1->SetXTitle("#epoch");
	PlotBkg_1->SetYTitle("tot loss");
	PlotBkg_1->Draw();

	lossVSEpoch_train->Draw("l same");
	lossVSEpoch_test->Draw("l same");

	TLegend* PlotLegend_1= new TLegend(0.6,0.7,0.7,0.8);
	PlotLegend_1->SetFillColor(0);
	PlotLegend_1->SetTextSize(0.045);
	PlotLegend_1->SetTextFont(52);
	PlotLegend_1->AddEntry(lossVSEpoch_train,"train","l");
	PlotLegend_1->AddEntry(lossVSEpoch_test,"test","l");
	PlotLegend_1->Draw("same");

	//- Classification loss
	Plot->cd(2);

	TH2D* PlotBkg_2= new TH2D("PlotBkg_2","",100,0,data->GetEntries()+1,100,loss_type_min-dloss_type,loss_type_max+dloss_type);
	PlotBkg_2->SetStats(0);	
	PlotBkg_2->SetXTitle("#epoch");
	PlotBkg_2->SetYTitle("classification loss");
	PlotBkg_2->Draw();

	lossTypeVSEpoch_train->Draw("l same");
	lossTypeVSEpoch_test->Draw("l same");

	TLegend* PlotLegend_2= new TLegend(0.6,0.7,0.7,0.8);
	PlotLegend_2->SetFillColor(0);
	PlotLegend_2->SetTextSize(0.045);
	PlotLegend_2->SetTextFont(52);
	PlotLegend_2->AddEntry(lossTypeVSEpoch_train,"train","l");
	PlotLegend_2->AddEntry(lossTypeVSEpoch_test,"test","l");
	PlotLegend_2->Draw("same");

	//- Regression loss
	Plot->cd(3);

	TH2D* PlotBkg_3= new TH2D("PlotBkg_3","",100,0,data->GetEntries()+1,100,loss_pars_min-dloss_pars,loss_pars_max+dloss_pars);
	PlotBkg_3->SetStats(0);	
	PlotBkg_3->SetXTitle("#epoch");
	PlotBkg_3->SetYTitle("regression loss");
	PlotBkg_3->Draw();

	lossParsVSEpoch_train->Draw("l same");
	lossParsVSEpoch_test->Draw("l same");

	TLegend* PlotLegend_3= new TLegend(0.6,0.7,0.7,0.8);
	PlotLegend_3->SetFillColor(0);
	PlotLegend_3->SetTextSize(0.045);
	PlotLegend_3->SetTextFont(52);
	PlotLegend_3->AddEntry(lossParsVSEpoch_train,"train","l");
	PlotLegend_3->AddEntry(lossParsVSEpoch_test,"test","l");
	PlotLegend_3->Draw("same");

}//close DrawNNLossVSEpoch()

