package HMM;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.util.ArrayList;

public class BrowWordProcess {
	public static int[] LoadSequence(String path)
	{
		ArrayList<Integer> list=new ArrayList<Integer>();
		DataInputStream in=null;
		try
		{
			in=new DataInputStream(new FileInputStream(path));
			String s=null;
			while((s=in.readLine())!=null)
			{
				s=s.toLowerCase();
				for(int i=0;i<s.length();++i)
				{
					char c=s.charAt(i);
					if((c<='z' && c>='a'))
					{
						list.add((int)(c-'a'));
					}
					else
					{
						list.add(26);//use blank instead of them
					}
				}
				list.add(26);
			}
			in.close();
			
			
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		int seq[]=new int[list.size()];
		for(int i=0;i<seq.length;++i)
		{
			seq[i]=list.get(i);
		}
		return seq;
	}

	public static ArrayList<int[]> LoadMultiSequence(String path,int SEQNUM)
	{
		ArrayList<int[]> seqList=new ArrayList<>();
		ArrayList<ArrayList<Integer>> list=new ArrayList<ArrayList<Integer>>();
		for(int i=0;i<SEQNUM;++i)
		{
			list.add(new ArrayList<Integer>());
		}
		DataInputStream in=null;
		int k=0;
		int listID=0;
		try
		{
			in=new DataInputStream(new FileInputStream(path));
			String s=null;
			while((s=in.readLine())!=null)
			{
				s=s.toLowerCase();
				listID=k%SEQNUM;
				++k;
				for(int i=0;i<s.length();++i)
				{
					char c=s.charAt(i);
					if((c<='z' && c>='a'))
					{
						
						list.get(listID).add((int)(c-'a'));
					}
					else
					{
						list.get(listID).add(26);//use blank instead of them
					}
				}
				list.get(listID).add(26);
			}
			in.close();
			
			
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		for(int s=0;s<SEQNUM;++s)
		{
			int seq[]=new int[list.get(s).size()];
			ArrayList<Integer> tmpList=list.get(s);
			for(int i=0;i<seq.length;++i)
			{
				seq[i]= tmpList.get(i);
			}
			seqList.add(seq);
		}
		return seqList;
	}

	public static ArrayList<int[]> LoadMultiSequence(String path)
	{
		ArrayList<int[]> seqList=new ArrayList<>();
		ArrayList<ArrayList<Integer>> list=new ArrayList<ArrayList<Integer>>();
		for(int i=0;i<26;++i)
		{
			list.add(new ArrayList<Integer>());
		}
		DataInputStream in=null;
		int k=0;
		int listID=0;
		try
		{
			in=new DataInputStream(new FileInputStream(path));
			String s=null;
			while((s=in.readLine())!=null)
			{
				s=s.toLowerCase();
				//listID=k%SEQNUM;
				char ch=s.charAt(0);
				if(ch<'a' || ch >'z')
				{
					continue;
				}
				listID=ch-'a';
				++k;
				for(int i=0;i<s.length();++i)
				{
					char c=s.charAt(i);
					if((c<='z' && c>='a'))
					{
						
						list.get(listID).add((int)(c-'a'));
					}
					else
					{
						list.get(listID).add(26);//use blank instead of them
					}
				}
				list.get(listID).add(26);
			}
			in.close();
			
			
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		for(int s=0;s<26;++s)
		{
			int seq[]=new int[list.get(s).size()];
			if(seq.length==0)
			{
				continue;
			}
			ArrayList<Integer> tmpList=list.get(s);
			for(int i=0;i<seq.length;++i)
			{
				seq[i]= tmpList.get(i);
			}
			seqList.add(seq);
		}
		return seqList;
	}

}
