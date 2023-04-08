using System;
using System.Text;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using static System.Console;
using static System.Math;
using System.Diagnostics;

public class Program{
    // 入力
    public static void Main(string[] args){
        ConsoleInput cin = new ConsoleInput(Console.In, ' ');

        new Program().Solve();
    }

    // ルール
    // - このプログラムは短時間で高速に動作する必要がある

    // 問題文
    // 

    // 制約
    // 

    // 以下に問題の具体的な解法をステップバイステップで書く

    public void Solve(){
        
    }
}

public class ConsoleInput{
    private readonly System.IO.TextReader _stream;
    private char _separator = ' ';
    private Queue<string> inputStream;
    public ConsoleInput(System.IO.TextReader stream, char separator = ' '){
        this._separator = separator;
        this._stream = stream;
        inputStream = new Queue<string>();
    }
    public string Read{
        get{
            if (inputStream.Count != 0) return inputStream.Dequeue();
            string[] tmp = _stream.ReadLine().Split(_separator);
            for (int i = 0; i < tmp.Length; ++i)
                inputStream.Enqueue(tmp[i]);
            return inputStream.Dequeue();
        }
    }

    // string型の入力を読み取る関数
    public string ReadLine { get { return _stream.ReadLine(); } }
    // int型の入力を読み取る関数
    public int ReadInt { get { return int.Parse(Read); } }
    // long型の入力を読み取る関数
    public long ReadLong { get { return long.Parse(Read); } }
    // double型の入力を読み取る関数
    public double ReadDouble { get { return double.Parse(Read); } }
    // string型の配列を読み取る関数
    public string[] ReadStrArray(long N) { var ret = new string[N]; for (long i = 0; i < N; ++i) ret[i] = Read; return ret;}
    // int型の配列を読み取る関数
    public int[] ReadIntArray(long N) { var ret = new int[N]; for (long i = 0; i < N; ++i) ret[i] = ReadInt; return ret;}
    // long型の配列を読み取る関数
    public long[] ReadLongArray(long N) { var ret = new long[N]; for (long i = 0; i < N; ++i) ret[i] = ReadLong; return ret;}

    // 配列を出力する関数
    public void PrintArray<T>(T[] array){Console.WriteLine(string.Join(" ", T));}

}