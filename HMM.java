package HMM;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Random;

import javax.jws.Oneway;

public class HMM {
private int m_hiddenStateNum=0;
private int m_observeNum=0;

private double []m_FI=null;
private double [][] m_transitionMatrix=null;
private double[][] m_confusionMatrix=null;

private int []m_stateCountArray=null; //use to change weight of current model
private int [][]m_transCountArray=null;
private int [][]m_confCountArray=null;

private static double LITTLE_BIAS=0.001;
private static double PROB_THRESHOLD=1E-20;

private boolean m_probNoZeroFlag=false;
private boolean m_alreadyTrainedOnce=false;

/**
 * Init the HMM
 * @param hiddenNum :the hidden state num;
 * @param obNum : the observation state num;
 */
public HMM(int hiddenNum,int obNum,boolean probNoZero)
{
	m_hiddenStateNum=hiddenNum;
	m_observeNum=obNum;
	m_transitionMatrix=new double[m_hiddenStateNum][m_hiddenStateNum];
	m_confusionMatrix=new double[m_hiddenStateNum][m_observeNum];
	m_FI=new double[m_hiddenStateNum];
	m_stateCountArray=new int[m_hiddenStateNum];
	m_transCountArray=new int[m_hiddenStateNum][m_hiddenStateNum];
	m_confCountArray=new int[m_hiddenStateNum][m_observeNum];
	m_probNoZeroFlag=probNoZero;
	m_alreadyTrainedOnce=false;
	//m_FI[0]=1;
	
}
/**
 * Init the parameter randomly ,but keep the constraint not be broken;
 * like: the sum of FI[j] be one;
 * 		 the sum of the transtion matrix from j to others will be one;
 * 		 the sum of the confusion matrix for state j emission observations;
 */
public void Init()
{
	Random r=new Random();
	double demonitor=0;
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		m_FI[i]=30+r.nextDouble()+LITTLE_BIAS/m_hiddenStateNum;
		demonitor+=m_FI[i];
	}
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		m_FI[i]/=demonitor;
	}
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		demonitor=0;
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			m_transitionMatrix[i][j]=30+r.nextDouble()+LITTLE_BIAS/m_hiddenStateNum;	
			demonitor+=m_transitionMatrix[i][j];
		}
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			m_transitionMatrix[i][j]/=demonitor;
		}
		demonitor=0;
		for(int k=0;k<m_observeNum;++k)
		{
			m_confusionMatrix[i][k]=30+r.nextDouble()+LITTLE_BIAS/m_observeNum;
			demonitor+=m_confusionMatrix[i][k];
		}
		for(int k=0;k<m_observeNum;++k)
		{
			m_confusionMatrix[i][k]/=demonitor;
		}
		
	}
}
/**
 * This function will use the forward algorithm to compute the partial alfa probabilitys;
 * @param obsSeq :the sequence used to compute the forward partial probabilitys; 
 * @param C :it will store the coefficient for re-estimate the back partial probs without low-overflow
 * @return	return the forward partial probs matrix;
 */
public double[][] ComputeForwardPartialAlfa(int []obsSeq,double []C) 
{
	double alfa[][]=new double[obsSeq.length][m_hiddenStateNum];
	C[0]=0;
	for(int j=0;j<m_hiddenStateNum;++j)
	{
		alfa[0][j]=m_FI[j]*m_confusionMatrix[j][obsSeq[0]];
		C[0]+=alfa[0][j];
	}
	C[0]=1/C[0];
	for(int j=0;j<m_hiddenStateNum;++j)
	{
		alfa[0][j]*=C[0];
	}
	for(int t=1;t<obsSeq.length;++t)
	{
		C[t]=0;
		for(int i=0;i<m_hiddenStateNum;++i)
		{
			double tmp=0.0;
			for(int j=0;j<m_hiddenStateNum;++j)
			{
				tmp+=alfa[t-1][j]*m_transitionMatrix[j][i];
			}
			tmp*=m_confusionMatrix[i][obsSeq[t]];
			alfa[t][i]=tmp;
			C[t]+=tmp;
		}
		C[t]=1/C[t];
		for(int i=0;i<m_hiddenStateNum;++i)
		{
			alfa[t][i]*=C[t];
		}
	}
	return alfa;
}

/**		
 * code by huangjin;
 * This function will use back forward algorithm to compute the backforward partial probs; 
 * @param obsSeq : which sequence will be used to compute the backforward partial information
 * @param C : this should be from the forward algorithm's C array;
 * @return the backforwar partial probs matrix
 */
double [][] ComputeBackPartiaBeta(int []obsSeq,double C[])
{
	int lastIndex=obsSeq.length-1;
	double beta[][]=new double[obsSeq.length][m_hiddenStateNum];
	for(int j=0;j<m_hiddenStateNum;++j)
	{
		beta[lastIndex][j]=C[lastIndex];
	}
	for(int t=lastIndex-1;t>=0;--t)
	{
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			double tmp=0;
			for(int jnext=0;jnext<m_hiddenStateNum;++jnext)
			{
				tmp+=m_transitionMatrix[j][jnext]*m_confusionMatrix[jnext][obsSeq[t+1]]*beta[t+1][jnext];
			}
			beta[t][j]=tmp*C[t];
		}
	}
	return beta;
}

/**
 * This use the veterbi algorithm to compute the hidden sequence which will be most happens;
 * @param obsSeq : wich observation sequence will be used to compute it's hidden sequence
 * @return	the hidden sequence;
 */
public int[] Veterbi(int []obsSeq)
{
	double theta[][]=new double[obsSeq.length][m_hiddenStateNum];
	int backPoint[][]=new int[obsSeq.length][m_hiddenStateNum]; // the backPoint[0][..] not use;
	for(int j=0;j<m_hiddenStateNum;++j)
	{
		//theta[0][j]=m_FI[j]*m_confusionMatrix[j][obsSeq[0]];
		theta[0][j]=Math.log(m_FI[j])+Math.log(m_confusionMatrix[j][obsSeq[0]]);
	}
	for(int t=1;t<obsSeq.length;++t)
	{
		for(int jcur=0;jcur<m_hiddenStateNum;++jcur)
		{
			double maxPartia=Math.log(m_transitionMatrix[0][jcur])+theta[t-1][0];;
			int bestBackPoint=0;
			double tmpPartia=0;
			for(int jprev=1;jprev<m_hiddenStateNum;++jprev)
			{
				//tmpPartia=theta[t-1][jprev]*m_transitionMatrix[jprev][jcur];;
				tmpPartia=Math.log(m_transitionMatrix[jprev][jcur])+theta[t-1][jprev];
				if(tmpPartia>maxPartia)
				{
					maxPartia=tmpPartia;
					bestBackPoint=jprev;
				}				
			}
			//maxPartia*=m_confusionMatrix[jcur][obsSeq[t]];
			maxPartia+=Math.log(m_confusionMatrix[jcur][obsSeq[t]]);
			theta[t][jcur]=maxPartia;
			backPoint[t][jcur]=bestBackPoint;
		}
	}
	int []hidSeq=new int[obsSeq.length];
	int bestJ=0;
	int t=obsSeq.length-1;
	double bestP=theta[t][0];
	for(int j=1;j<m_hiddenStateNum;++j)
	{
		if(theta[t][j]>bestP)
		{
			bestJ=j;
			bestP=theta[t][j];
		}
	}
	hidSeq[t]=bestJ;
	for(;t>0;--t)
	{
		hidSeq[t-1]=backPoint[t][hidSeq[t]];
	}
	return hidSeq;
}

/**
 * this function will use the forward algorithm to compute the prob of the sequece using this model 
 * @param obsSequece :the observed state sequece to find the prob generated by the model
 * 					every element int the array is an index to stand for the observe state
 * @return the probability of the model to generate the sequence, it's a log prob
 */
public double ComputeProb(int []obsSequece)
{
	double C[]=new double[obsSequece.length];
	double [][]alfa=ComputeForwardPartialAlfa(obsSequece,C);
	double prob=0;
	for(int t=0;t<obsSequece.length;++t)
	{
		prob+=Math.log(C[t]);
	}
	return -prob;
}
/**
 * Use one sequence to train the HMM,using forward and backforward (Baum-Welch) algorithm;
 * @param obsSeq : the training observation sequence;
 * @param maxIterate : the max epoches will the training do;
 */
public void TrainHMMWithForwardAndBackMethod(int obsSeq[],int maxIterate)
{
	double gama[][]=new double[obsSeq.length-1][m_hiddenStateNum];
	double sigma[][][]=new double[obsSeq.length-1][m_hiddenStateNum][m_hiddenStateNum];
	
	double C[]=new double[obsSeq.length];
	double alfa[][]=ComputeForwardPartialAlfa(obsSeq,C);
	double beta[][]=ComputeBackPartiaBeta(obsSeq,C);
	
	double oldLogProb=0; 
	double curLogProb=0;
	int lastIndex=obsSeq.length-1;
	for(int t=0;t<obsSeq.length;++t)
	{
		oldLogProb+=Math.log(C[t]);
	}
	oldLogProb*=(-1);
	int iterate=0;
	do{
		
		int end=obsSeq.length-1;
		for(int t=0;t<end;++t)
		{
			double demonitor=0;
			for(int i=0;i<m_hiddenStateNum;++i)
			{
				for(int j=0;j<m_hiddenStateNum;++j)
				{
					demonitor+=alfa[t][i]*m_transitionMatrix[i][j]*m_confusionMatrix[j][obsSeq[t+1]]*beta[t+1][j];
				}
			}
			for(int i=0;i<m_hiddenStateNum;++i)
			{
				gama[t][i]=0;
				for(int j=0;j<m_hiddenStateNum;++j)
				{
					sigma[t][i][j]=alfa[t][i]*m_transitionMatrix[i][j]*m_confusionMatrix[j][obsSeq[t+1]]*beta[t+1][j]/demonitor;
					gama[t][i]+=sigma[t][i][j];
				}
			}
		}
		
		//recalculate FI
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			if(m_probNoZeroFlag)
			{
				m_FI[j]=LITTLE_BIAS/(m_hiddenStateNum)+(1-LITTLE_BIAS)*gama[0][j];
			}
			else
			{
				m_FI[j]=gama[0][j];
			}
		}
		//recalculate transition matrix
		for(int i=0;i<m_hiddenStateNum;++i)
		{
			for(int j=0;j<m_hiddenStateNum;++j)
			{
				double tmpSigma=0;
				double tmpGama=0;
				for(int t=0;t<end;++t)
				{
					tmpSigma+=sigma[t][i][j];
					tmpGama+=gama[t][i];
				}
				if(m_probNoZeroFlag)
				{
					m_transitionMatrix[i][j]=LITTLE_BIAS/(m_hiddenStateNum)+(1-LITTLE_BIAS)*(tmpSigma/tmpGama);
				}
				else
				{
					m_transitionMatrix[i][j]=(tmpSigma/tmpGama);
				}
			}
		}
		//recalculate confusion matrix
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			for(int k=0;k<m_observeNum;++k)
			{
				double sum=0;
				double kP=0;
				for(int t=0;t<lastIndex;++t)
				{
					if(obsSeq[t]==k)
					{
						kP+=gama[t][j];
					}
					sum+=gama[t][j];
				}
				if(m_probNoZeroFlag)
				{
					m_confusionMatrix[j][k]=LITTLE_BIAS/(m_observeNum)+(1-LITTLE_BIAS)*(kP/sum);
				}
				else
				{
					m_confusionMatrix[j][k]=(kP/sum);
				}
			}
		}
		alfa=ComputeForwardPartialAlfa(obsSeq,C);
		beta=ComputeBackPartiaBeta(obsSeq,C);
		
		curLogProb=0;
		for(int t=0;t<obsSeq.length;++t)
		{
			curLogProb+=Math.log(C[t]);
		}
		curLogProb*=(-1);
		double p=Math.abs(curLogProb/oldLogProb);
		/*if(p>0.99999999999999&&p<1.0000000000001)
		{
			break;
		}*/
		System.out.println("iterate:"+iterate+"\trate:"+p+"\tcurrate:"+curLogProb);
		
		if(iterate>maxIterate )//|| curLogProb<=oldLogProb )
		{
			break;
		}
		oldLogProb=curLogProb;
		++iterate;
	}while(true);	
	int hidSeq[]=Veterbi(obsSeq);
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		m_stateCountArray[i]=0;
	}
	for(int i=0;i<hidSeq.length;++i)
	{
		++m_stateCountArray[hidSeq[i]]; //compute the obseq state count information
	}
	
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			m_transCountArray[i][j]=0;
		}
	}
	for(int i=0;i<hidSeq.length-1;++i)
	{
		m_transCountArray[hidSeq[i]][hidSeq[i+1]]++;
	}
	
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		for(int j=0;j<m_observeNum;++j)
		{
			m_confCountArray[i][j]=0;
		}
	}
	for(int i=0;i<obsSeq.length;++i)
	{
		m_confCountArray[hidSeq[i]][obsSeq[i]]++;
	}
	m_alreadyTrainedOnce=true;
}

public HMM copyHMM()
{
	HMM hmm=new HMM(m_hiddenStateNum,m_observeNum,m_probNoZeroFlag);
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		hmm.m_FI[i]=m_FI[i];
		hmm.m_stateCountArray[i]=m_stateCountArray[i];
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			hmm.m_transitionMatrix[i][j]=m_transitionMatrix[i][j];
			hmm.m_transCountArray[i][j]=m_transCountArray[i][j];
		}
	}
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		for(int k=0;k<m_observeNum;++k)
		{
			hmm.m_confusionMatrix[i][k]=m_confusionMatrix[i][k];
			hmm.m_confCountArray[i][k]=m_confCountArray[i][k];
		}
	}
	hmm.m_alreadyTrainedOnce=m_alreadyTrainedOnce;
	return hmm;
}

private double getWhichStateCount(int i)
{
	return m_stateCountArray[i];
}
public double getWhichFI(int i)
{
	return m_FI[i];
}
public double getWhichTransParamter(int i, int j)
{
	return m_transitionMatrix[i][j];
}
public double getWhichConfsParameter(int i, int obs)
{
	return m_confusionMatrix[i][obs];
}
public int getHiddenStateNum()
{
	return m_hiddenStateNum;
}
public int getObserveNum()
{
	return m_observeNum;
}
/**
 * @return if the model has been trained,if the param in the model already be trained,return true;else return false;
 */
public boolean getAlreadyTrainedOnceFlag()
{
	return m_alreadyTrainedOnce;
}
/**
 * this function use to set the model's flag ,if the model trained already,you should firstly set this flag;
 * this flag will be used for dynamic learning method for future sequence 
 * @param flag
 */
public void setAlreadyTrainedOnceFlag(boolean flag)
{
	m_alreadyTrainedOnce=flag;
}

/**
 * This function will combine these models to a single model,each model have the sequence hidden state information;
 * @param hmmList : a list contains the models which will be combined to a one;
 * 					their paramters must have the same range
 * @return a new HMM which combined multi-hmm;
 */
public HMM CombineHMM(ArrayList<HMM> hmmList)
{
	if(hmmList==null ||hmmList.isEmpty())
	{
		System.out.println("paramter of CombineHMM is not right");
		return null;
	}
	HMM firstHMM=hmmList.get(0);
	int len=hmmList.size();
	
	int hiddenStateNum=firstHMM.m_hiddenStateNum;
	int observeStateNum=firstHMM.m_observeNum;
	
	HMM resHMM=new HMM(hiddenStateNum,firstHMM.m_observeNum,m_probNoZeroFlag);
	double weight[][]=new double[hmmList.size()][hiddenStateNum];
	for(int j=0;j<hiddenStateNum;++j)
	{
		double sum=0;
		
		for(int s=0;s<len;++s)
		{
			weight[s][j]=hmmList.get(s).getWhichStateCount(j);
			sum+=weight[s][j];
		}
		for(int s=0;s<len;++s)
		{
			weight[s][j]/=sum;
		}
	}
	
	//combine FI information;
	double sumFI=0;
	for(int j=0;j<hiddenStateNum;++j)
	{
		double m=0;
		for(int s=0;s<len;++s)
		{
			m+=weight[s][j]*hmmList.get(s).getWhichFI(j);
		}
		resHMM.m_FI[j]=m;
		sumFI+=m;
	}
	
	// combine transtion matrix paramter;
	for(int i=0;i<hiddenStateNum;++i)
	{
		double sumT=0;
		for(int j=0;j<hiddenStateNum;++j)
		{
			double m=0;
			for(int s=0;s<len;++s)
			{
				m+=weight[s][i]*hmmList.get(s).getWhichTransParamter(i, j);
			}
			resHMM.m_transitionMatrix[i][j]=m;
			sumT+=m;
		}
		for(int j=0;j<hiddenStateNum;++j)
		{
			resHMM.m_transitionMatrix[i][j]/=sumT;
		}
	}
	//combine confusion matrix
	for(int i=0;i<hiddenStateNum;++i)
	{
		double sumC=0;
		for(int j=0;j<observeStateNum;++j)
		{
			double m=0;
			for(int s=0;s<len;++s)
			{
				m+=weight[s][i]*hmmList.get(s).getWhichConfsParameter(i, j);
			}
			resHMM.m_confusionMatrix[i][j]=m;
			sumC+=m;
		}
		for(int j=0;j<observeStateNum;++j)
		{
			resHMM.m_confusionMatrix[i][j]/=sumC;
		}
	}
	return resHMM;
}

/**
 * This function will use every sequence to train a model ,and combine these model ,and then continue to train multi-models...
 * @param seqList : a list contains some sequence need to train a combined model
 * @param combineMaxIterate : combined iterate numbers;
 * @param trainHMMMaxIterate : max iterate time when a single model trained
 */
public void TrainHMMWithCombineHMMMethod(ArrayList<int[]> seqList,int combineMaxIterate,int trainHMMMaxIterate)
{
//	int iterate=0;
//	ArrayList<HMM> hmmList=new ArrayList<>();
//	while(iterate<maxIterate)
//	{
//		++iterate;
//		int len=seqList.size();
//		hmmList.clear();
//		for(int i=0;i<len;++i)
//		{
//			HMM hmm=TrainNewHMMOnceWithNoChange2CurrentHMM(seqList.get(i));
//			hmmList.add(hmm);
//		}
//		if(iterate==1)
//		{
//			System.out.println(iterate);
//		}
//		HMM comHMM=CombineHMM(hmmList);
//		this.m_confusionMatrix=comHMM.m_confusionMatrix;
//		this.m_FI=comHMM.m_FI;
//		this.m_transitionMatrix=comHMM.m_transitionMatrix;
//		System.out.println("iterate :"+iterate);
//	}
	int iterate=0;
	int len=seqList.size();
	ArrayList<HMM> hmmList=new ArrayList<>();
	for(int i=0;i<len;++i)
	{
		HMM hmm=copyHMM();
		hmmList.add(hmm);
	}
	while(iterate<combineMaxIterate)
	{
		++iterate;
		for(int i=0;i<len;++i)
		{
			hmmList.get(i).TrainHMMWithForwardAndBackMethod(seqList.get(i), trainHMMMaxIterate);
		}
		if(iterate==1)
		{
			System.out.println(iterate);
		}
		HMM comHMM=CombineHMM(hmmList);
		this.m_confusionMatrix=comHMM.m_confusionMatrix;
		this.m_FI=comHMM.m_FI;
		this.m_transitionMatrix=comHMM.m_transitionMatrix;
		System.out.println("iterate :"+iterate);
		hmmList.clear();
		for(int i=0;i<len;++i)
		{
			HMM hmm=copyHMM();
			hmmList.add(hmm);
		}
	}
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			m_transCountArray[i][j]=0;
		}
	}
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		for(int j=0;j<m_observeNum;++j)
		{
			m_confCountArray[i][j]=0;
		}
	}
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		m_stateCountArray[i]=0;
	}
	
	for(int s=0;s<seqList.size();++s)
	{
		int []obsSeq=seqList.get(s);
		int hidSeq[]=Veterbi(obsSeq);
		for(int i=0;i<hidSeq.length-1;++i)
		{
			m_transCountArray[hidSeq[i]][hidSeq[i+1]]++;
		}
		for(int i=0;i<obsSeq.length;++i)
		{
			m_confCountArray[hidSeq[i]][obsSeq[i]]++;
		}
		for(int i=0;i<hidSeq.length;++i)
		{
			++m_stateCountArray[hidSeq[i]]; //compute the obseq state count information
		}
	}
	m_alreadyTrainedOnce=true;
}

/**
 * this function differ from the HMM combined method, the sequence list will be used together to train a model
 * @param seqList: the sequence list
 * @param maxIterate:	the max iterate time;
 */
public void TrainHMMWithMultiSequenceTogetherMethod(ArrayList<int[]> seqList,int maxIterate)
{
	int seqLen=seqList.size();
	ArrayList<double[][]>gamaList=new ArrayList<double[][]>();
	ArrayList<double[][][]>sigmaList=new ArrayList<double[][][]>();
	ArrayList<double []>cList=new ArrayList<double[]>();
	ArrayList<double[][]> alfaList=new ArrayList<>();
	ArrayList<double[][]> betaList=new ArrayList<>();
	
	for(int i=0;i<seqLen;++i)
	{
		int seq[]=seqList.get(i);
		gamaList.add(new double[seq.length-1][m_hiddenStateNum]);
		sigmaList.add(new double[seq.length-1][m_hiddenStateNum][m_hiddenStateNum]);
		cList.add(new double[seq.length]);
		alfaList.add(ComputeForwardPartialAlfa(seq, cList.get(i)));
		betaList.add(ComputeBackPartiaBeta(seq, cList.get(i)));
	}
	
	
	double oldLogProb=0; 
	double curLogProb=0;
	for(int i=0;i<seqLen;++i)
	{
		double C[]=cList.get(i);
		for(int t=0;t<seqList.get(i).length;++t)
		{
			oldLogProb+=Math.log(C[t]);
		}
	}
	oldLogProb*=(-1);
	int iterate=0;
	do{
		for(int s=0;s<seqLen;++s)
		{
			int []obsSeq=seqList.get(s);
			double alfa[][]=alfaList.get(s);
			double gama[][]= gamaList.get(s);
			double sigma[][][]=sigmaList.get(s);
			double beta[][]=betaList.get(s);
			double C[]=cList.get(s);
			int end=obsSeq.length-1;
			for(int t=0;t<end;++t)
			{	
				for(int i=0;i<m_hiddenStateNum;++i)
				{
					gama[t][i]=0;
					for(int j=0;j<m_hiddenStateNum;++j)
					{
						sigma[t][i][j]=alfa[t][i]*m_transitionMatrix[i][j]*m_confusionMatrix[j][obsSeq[t+1]]*beta[t+1][j];
					}
					gama[t][i]=alfa[t][i]*beta[t][i]/C[t];
				}
			}
		}
		//recalculate FI
		double sumFI=0;
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			double g=0;
			for(int s=0;s<seqLen;++s)
			{
				double gama[][]=gamaList.get(s);
				g+=gama[0][j];
				
			}
			if(m_probNoZeroFlag)
			{
				m_FI[j]=LITTLE_BIAS/(m_hiddenStateNum)+(1-LITTLE_BIAS)*g;
			}
			else
			{
				m_FI[j]=g;
			}
			sumFI+=m_FI[j];
		}
		
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			m_FI[j]/=sumFI;
		}
		//recalculate transition matrix
		for(int i=0;i<m_hiddenStateNum;++i)
		{
			for(int j=0;j<m_hiddenStateNum;++j)
			{
				double tmpSigma=0;
				double tmpGama=0;
				for(int s=0;s<seqLen;++s)
				{
					int end=seqList.get(s).length-1;
					double sigma[][][]=sigmaList.get(s);
					double gama[][]=gamaList.get(s);
					for(int t=0;t<end;++t)
					{
						tmpSigma+=sigma[t][i][j];
						tmpGama+=gama[t][i];
					}
				}
				if(m_probNoZeroFlag)
				{
					m_transitionMatrix[i][j]=LITTLE_BIAS/(m_hiddenStateNum)+(1-LITTLE_BIAS)*(tmpSigma/tmpGama);
				}
				else
				{
					m_transitionMatrix[i][j]=(tmpSigma/tmpGama);
				}
			}
		}
		//recalculate confusion matrix
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			for(int k=0;k<m_observeNum;++k)
			{
				double sum=0;
				double kP=0;
				for(int s=0;s<seqLen;++s)
				{
					
					int []obsSeq=seqList.get(s);
					int end=obsSeq.length-1;
					double gama[][]=gamaList.get(s);
					for(int t=0;t<end;++t)
					{
						if(obsSeq[t]==k)
						{
							kP+=gama[t][j];
						}
						sum+=gama[t][j];
					}
				}
				if(m_probNoZeroFlag)
				{
					m_confusionMatrix[j][k]=LITTLE_BIAS/(m_observeNum)+(1-LITTLE_BIAS)*(kP/sum);
				}
				else
				{
					m_confusionMatrix[j][k]=(kP/sum);
				}
			}
		}

		alfaList.clear();
		betaList.clear();
		for(int i=0;i<seqLen;++i)
		{
			int seq[]=seqList.get(i);
			alfaList.add(ComputeForwardPartialAlfa(seq, cList.get(i)));
			betaList.add(ComputeBackPartiaBeta(seq, cList.get(i)));
		}
		curLogProb=0;
		for(int i=0;i<seqLen;++i)
		{
			double C[]=cList.get(i);
			for(int t=0;t<seqList.get(i).length;++t)
			{
				curLogProb+=Math.log(C[t]);
			}
		}
		curLogProb*=(-1);
		double p=Math.abs(curLogProb/oldLogProb);
		/*if(p>0.99999999999999&&p<1.0000000000001)
		{
			break;
		}*/
		System.out.println("iterate:"+iterate+"\trate:"+p+"\tcurrate:"+curLogProb);
		
		if(iterate>maxIterate)// || curLogProb<=oldLogProb )
		{
			break;
		}
		oldLogProb=curLogProb;
		++iterate;
	}while(true);	
	
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			m_transCountArray[i][j]=0;
		}
	}
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		for(int j=0;j<m_observeNum;++j)
		{
			m_confCountArray[i][j]=0;
		}
	}
	for(int i=0;i<m_hiddenStateNum;++i)
	{
		m_stateCountArray[i]=0;
	}
	
	for(int s=0;s<seqList.size();++s)
	{
		int []obsSeq=seqList.get(s);
		int hidSeq[]=Veterbi(obsSeq);
		for(int i=0;i<hidSeq.length-1;++i)
		{
			m_transCountArray[hidSeq[i]][hidSeq[i+1]]++;
		}
		for(int i=0;i<obsSeq.length;++i)
		{
			m_confCountArray[hidSeq[i]][obsSeq[i]]++;
		}
		for(int i=0;i<hidSeq.length;++i)
		{
			++m_stateCountArray[hidSeq[i]]; //compute the obseq state count information
		}
	}
	m_alreadyTrainedOnce=true;
}

public void TrainHMMDynamicLearningMethod(int[] obsSeq,int maxIterate) throws HMMException
{
	if(!m_alreadyTrainedOnce)
	{
		throw new HMMException("the model not already be trained,please make sure,and use the method setAlreadyTrainedOnceFlag(boolean)");
	}
	else
	{
		HMM newHMM=copyHMM();
		newHMM.TrainHMMWithForwardAndBackMethod(obsSeq, maxIterate);
		int []stateCountA=m_stateCountArray;
		int [][]transCountA=m_transCountArray;
		int [][]confCountA=m_confCountArray;
		
		int []stateCountB=newHMM.m_stateCountArray;
		int [][]transCountB=newHMM.m_transCountArray;
		int [][]confCountB=newHMM.m_confCountArray;
		double weight=0.5;
		for(int i=0;i<m_hiddenStateNum;++i)
		{
			double totalStateCount=stateCountA[i]+stateCountB[i];
			weight=stateCountB[i]/totalStateCount;
			m_FI[i]=m_FI[i]*(1-weight)+newHMM.m_FI[i]*weight;
			for(int j=0;j<m_hiddenStateNum;++j)
			{
				m_transitionMatrix[i][j]=(transCountA[i][j]+transCountB[i][j])/totalStateCount;
			}
			for(int k=0;k<m_observeNum;++k)
			{
				m_confusionMatrix[i][k]=(confCountA[i][k]+confCountB[i][k])/totalStateCount;
			}
		}
	}
}

public static HMM CombineHMMByModuleMethod(ArrayList<HMM> hmmList) throws HMMException
{
	int hmmNum=hmmList.size();
	boolean checkFlag=true;
	if(hmmNum==0)
	{
		return null;
	}
	for(int i=0;i<hmmNum;++i)
	{
		HMM hmm=hmmList.get(i);
		if(!hmm.getAlreadyTrainedOnceFlag())
		{
			checkFlag=false;
			throw new HMMException("the "+i+" hmm in CombineHMMByModuleMethod not be trained yet!");
		}
	}
	for(int i=0;i<hmmNum-1;++i)
	{
		HMM hmmFirst=hmmList.get(i);
		HMM hmmSecond=hmmList.get(i+1);
		if(hmmFirst.getHiddenStateNum()!=hmmSecond.getHiddenStateNum())
		{
			checkFlag=false;
			throw new HMMException("the hmms have difference hidden state num!");
		}
		if(hmmFirst.getObserveNum()!=hmmSecond.getObserveNum())
		{
			checkFlag=false;
			throw new HMMException("the hmms have difference observe state num!");
		}
		
	}
	if(!checkFlag)
	{
		return null;
	}
	int hidNum=hmmList.get(0).getHiddenStateNum();
	int obsNum=hmmList.get(0).getObserveNum();
	HMM hmm=new HMM(hidNum,obsNum,true);
	for(int i=0;i<hidNum;++i)
	{
		hmm.m_stateCountArray[i]=0; 
		hmm.m_FI[i]=0;
		for(int m=0;m<hmmNum;++m)
		{
			HMM tmpHMM=hmmList.get(m);
			hmm.m_stateCountArray[i]+=tmpHMM.m_stateCountArray[i];
			hmm.m_FI[i]+=tmpHMM.getWhichFI(i)*tmpHMM.m_stateCountArray[i];
		}
		hmm.m_FI[i]/=hmm.m_stateCountArray[i];
		
		for(int j=0;j<hidNum;++j)
		{
			int tranCount=0;
			double stateCount=0;
			for(int m=0;m<hmmNum;++m)
			{
				HMM tmpHMM=hmmList.get(m);
				tranCount+=tmpHMM.m_transCountArray[i][j];
				stateCount+=tmpHMM.m_stateCountArray[i];
			}
			hmm.m_transitionMatrix[i][j]=tranCount/stateCount;
			hmm.m_transCountArray[i][j]=tranCount;
		}
		for(int j=0;j<obsNum;++j)
		{
			int confCount=0;
			double stateCount=0;
			for(int m=0;m<hmmNum;++m)
			{
				HMM tmpHMM=hmmList.get(m);
				confCount+=tmpHMM.m_confCountArray[i][j];
				stateCount+=tmpHMM.m_stateCountArray[i];
			}
			hmm.m_confusionMatrix[i][j]=confCount/stateCount;
			hmm.m_confCountArray[i][j]=confCount;
		}
	}
	return hmm;
}

public int[] MapStringToIntArray(String s)
{
	int a[]=new int[s.length()];
	for(int i=0;i<a.length;++i)
	{
		char c=s.charAt(i);
		if(c>='a' && c <='z')
		{
			a[i]=(int)(c-'a');
		}
		else
		{
			a[i]=26;
		}
	}
	return a;
}
public String MapIntArrayToString(int []a)
{
	StringBuilder sb=new StringBuilder();
	for(int i=0;i<a.length;++i)
	{
		if(a[i]<26)
		{
			sb.append((char)(a[i]+'a'));
		}
		else
		{
			sb.append(' ');
		}
	}
	return sb.toString();
}
public void SaveParameterFiles(String fiFile,String tranFile,String confFile)
{
	FileOutputStream outConf=null;
	FileOutputStream outTran=null;
	FileOutputStream outFI=null;
	try
	{
		
		outConf=new FileOutputStream(confFile);
		outTran=new FileOutputStream(tranFile);
		outFI=new FileOutputStream(fiFile);
		
		for(int i=0;i<26;++i)
		{
			char c=(char)(i+'a');
			StringBuilder sb=new StringBuilder();
			sb.append(c+"\t");
			for(int j=0;j<m_hiddenStateNum;++j)
			{
				sb.append(m_confusionMatrix[j][i]);
				sb.append("\t");
			}

			sb.append("\n");
			outConf.write(sb.toString().getBytes());
		}
		StringBuilder sb=new StringBuilder();
		sb.append("blank\t");
		for(int j=0;j<m_hiddenStateNum;++j)
		{
			sb.append(m_confusionMatrix[j][26]);
			sb.append("\t");
		}
		sb.append("\n");
		outConf.write(sb.toString().getBytes());
		outConf.flush();
		outConf.close();
		
		for(int i=0;i<m_hiddenStateNum;++i)
		{
			sb=new StringBuilder();
			for(int j=0;j<m_hiddenStateNum;++j)
			{
				sb.append(m_transitionMatrix[i][j]);
				sb.append("\t");
			}
			sb.append("\n");
			outTran.write(sb.toString().getBytes());
		}
		
		sb=new StringBuilder();
		for(int i=0;i<m_hiddenStateNum;++i)
		{
			sb.append(m_FI[i]+"\t");
		}
		outFI.write(sb.toString().getBytes());
		
	}catch(Exception e)
	{
		e.printStackTrace();
	}
	
}

public static void main(String args[])
{
	String initFIFile="D:/java_work_space/NeuralNetwork/HMM_data/initFI.txt";
	String initTranFile="D:/java_work_space/NeuralNetwork/HMM_data/initTran.txt";
	String initConfFile="D:/java_work_space/NeuralNetwork/HMM_data/initConf.txt";
	
	String fileConf="D:/java_work_space/NeuralNetwork/HMM_data/confusion.txt";
	String fileTran="D:/java_work_space/NeuralNetwork/HMM_data/tran.txt";
	String fileFI="D:/java_work_space/NeuralNetwork/HMM_data/fi.txt";
	
	String dataFile="D:/java_work_space/NeuralNetwork/HMM_data/words.txt";
	int []seq=BrowWordProcess.LoadSequence(dataFile);
	int SEQ_NUM=5;
	int AVG_NUM=seq.length/5;
	ArrayList<int[]> seqList=new ArrayList<>();
//	seqList=BrowWordProcess.LoadMultiSequence(dataFile, SEQ_NUM);
	seqList=BrowWordProcess.LoadMultiSequence(dataFile);
	int maxIterate=200;
	HMM hmm=new HMM(2,27,true);
	hmm.Init();
	hmm.SaveParameterFiles(initFIFile,initTranFile,initConfFile);
//	hmm.TrainHMMWithForwardAndBackMethod(seq,maxIterate);

//	hmm.TrainHMMWithMultiSequenceTogetherMethod(seqList, maxIterate);
	
//	hmm.TrainHMMWithCombineHMMMethod(seqList,5,maxIterate);
	
//	int laterSeq[]=hmm.MapStringToIntArray("this is a test");
//	try {
//		hmm.TrainHMMDynamicLearningMethod(laterSeq, maxIterate);
//	} catch (HMMException e) {
//		// TODO Auto-generated catch block
//		e.printStackTrace();
//	}
//	
	ArrayList<HMM> hmmList=new ArrayList<>();
	for(int i=0;i<seqList.size();++i)
	{
		HMM hmmH=hmm.copyHMM();
		hmmH.Init();
		hmmH.TrainHMMWithForwardAndBackMethod(seq,400);
		hmmList.add(hmmH);
	}
	try {
		hmm=CombineHMMByModuleMethod(hmmList);
	} catch (HMMException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
//	
	hmm.SaveParameterFiles(fileFI, fileTran, fileConf);
	
	int test[]=hmm.MapStringToIntArray("fiv me a word");
	int hid[]=hmm.Veterbi(test);
	for(int i=0;i<hid.length;++i)
	{
		System.out.print(hid[i]+"\t");
	}
	System.out.println();
	
	
	
	//format the paramter's output
	
//	FileOutputStream outConf=null;
//	FileOutputStream outTran=null;
//	try
//	{
//		
//		outConf=new FileOutputStream(fileConf);
//		outTran=new FileOutputStream(fileTran);
//		for(int i=0;i<26;++i)
//		{
//			char c=(char)(i+'a');
//			StringBuilder sb=new StringBuilder();
//			sb.append(c+"\t");
//			for(int j=0;j<hmm.m_hiddenStateNum;++j)
//			{
//				sb.append(hmm.m_confusionMatrix[j][i]);
//				sb.append("\t");
//			}
//
//			sb.append("\n");
//			outConf.write(sb.toString().getBytes());
//		}
//		StringBuilder sb=new StringBuilder();
//		sb.append("blank\t");
//		for(int j=0;j<hmm.m_hiddenStateNum;++j)
//		{
//			sb.append(hmm.m_confusionMatrix[j][26]);
//			sb.append("\t");
//		}
//		sb.append("\n");
//		outConf.write(sb.toString().getBytes());
//		outConf.flush();
//		outConf.close();
//		
//		for(int i=0;i<hmm.m_hiddenStateNum;++i)
//		{
//			sb=new StringBuilder();
//			for(int j=0;j<hmm.m_hiddenStateNum;++j)
//			{
//				sb.append(hmm.m_transitionMatrix[i][j]);
//				sb.append("\t");
//			}
//			sb.append("\n");
//			outTran.write(sb.toString().getBytes());
//		}
//		
//	}catch(Exception e)
//	{
//		e.printStackTrace();
//	}
	
	/*
	int obs[]=new int[]{
			6,9,4,7,4,2,3,
			5,1,0,2,0,9,3,
			6,6,8,8,4,5,9,
			2,7};
	int obs[]=new int[]{
			9,2,4,5,8,8,8,
			4,7,7,7,2,1,6,
			2,7,4,9,3,7,5,
			6,9,4,7,4,2,3,
			5,1,0,2,0,9,3,
			6,6,8,8,4,5,9,
			2,7,1,3,0,3,7,
			8,7,3,0,1,0,1,
			9,8,9,7,1,9,8,
			9,9,4,4,5,5,8,
			7,4,8,7,2,9,5,
			8,1,1,9,3,4,9,
			7,3,1,9,4,4,8,
			9,8,9,1,5,3,9,
			5,4,6,1,1,3,6,
			1,4,1,4,1,4,8,
			8,7,0,0,3,9,3,
			5,2,2,4,5,9,6
			};
	hmm.ForwardAndBackEM(obs);
	int test[]=new int[]{6,9,4,7,4,2,3,
			5,1,0,2,0,9,3,
			6,6,8,8,4,5,9,
			2,7};
	double r=hmm.ComputeProb(test);
	
	System.out.println("rate:"+r);
	*/
}


}
