## VPYLM

C++による可変長n-gram言語モデル（VPYLM）の実装です。

以下の論文をもとに実装を行っています。

- [Pitman-Yor 過程に基づく可変長 n-gram 言語モデル](http://chasen.org/~daiti-m/paper/nl178vpylm.pdf)

素組みしただけの愚直な実装ですので高速化などは考えていません。

[この記事](http://musyoku.github.io/2016/07/28/Pitman-Yor%E9%81%8E%E7%A8%8B%E3%81%AB%E5%9F%BA%E3%81%A5%E3%81%8F%E5%8F%AF%E5%A4%89%E9%95%B7n-gram%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB/)で実装したプログラムになります。

Pythonラッパーもあります。

- [python-vpylm](https://github.com/musyoku/vpylm-python)

## 動作環境

- C++11
- Boost 1.6

この実装は以下のコードを含んでいます。

- [c_printf](https://github.com/Gioyik/c_printf)

## 実行

### コンパイル

```
make vpylm
```

### 学習

実行前に学習用テキストデータを`train.txt`に保存し、適当なディレクトリに入れておきます。

またモデル保存用のディレクトリ`model`もあらかじめ作成しておいてください。

実行は

```
./vpylm -t alice/train.txt
```

のように行います。

### 結果

```
VPYLMを初期化しています ...
G0 <- 0.000343
Epoch 1 / 100 - 12486.1 lps - 1474.198 ppl - 6 depth - 1615 nodes - 38000 customers
Epoch 2 / 100 - 10963.4 lps - 1096.354 ppl - 6 depth - 2748 nodes - 39516 customers
Epoch 3 / 100 - 9988.9 lps - 913.213 ppl - 5 depth - 3467 nodes - 40673 customers
Epoch 4 / 100 - 9173.5 lps - 806.970 ppl - 7 depth - 4153 nodes - 41564 customers
Epoch 5 / 100 - 9173.5 lps - 709.064 ppl - 6 depth - 4712 nodes - 42388 customers
Epoch 6 / 100 - 9173.5 lps - 645.929 ppl - 8 depth - 5235 nodes - 43145 customers
Epoch 7 / 100 - 8813.7 lps - 588.926 ppl - 6 depth - 5503 nodes - 43619 customers
Epoch 8 / 100 - 8481.1 lps - 541.794 ppl - 8 depth - 5917 nodes - 44274 customers
Epoch 9 / 100 - 8324.1 lps - 494.373 ppl - 7 depth - 6304 nodes - 45014 customers
Epoch 10 / 100 - 8172.7 lps - 451.457 ppl - 6 depth - 6720 nodes - 45773 customers
Epoch 11 / 100 - 8026.8 lps - 396.790 ppl - 7 depth - 7193 nodes - 46696 customers
Epoch 12 / 100 - 7618.6 lps - 359.356 ppl - 6 depth - 7400 nodes - 47266 customers
Epoch 13 / 100 - 7618.6 lps - 328.192 ppl - 8 depth - 7671 nodes - 47951 customers
Epoch 14 / 100 - 7023.4 lps - 290.340 ppl - 7 depth - 8025 nodes - 48873 customers
Epoch 15 / 100 - 7308.9 lps - 261.595 ppl - 7 depth - 8242 nodes - 49727 customers
Epoch 16 / 100 - 6915.4 lps - 240.355 ppl - 7 depth - 8466 nodes - 50408 customers
Epoch 17 / 100 - 6969.0 lps - 213.343 ppl - 7 depth - 8651 nodes - 51232 customers
Epoch 18 / 100 - 6969.0 lps - 191.334 ppl - 7 depth - 9104 nodes - 52284 customers
Epoch 19 / 100 - 6862.6 lps - 187.046 ppl - 8 depth - 9308 nodes - 52885 customers
Epoch 20 / 100 - 6610.3 lps - 169.193 ppl - 8 depth - 9635 nodes - 53897 customers
Epoch 21 / 100 - 6514.5 lps - 163.198 ppl - 7 depth - 9907 nodes - 54394 customers
Epoch 22 / 100 - 6200.0 lps - 155.522 ppl - 8 depth - 10141 nodes - 55142 customers
Epoch 23 / 100 - 4994.4 lps - 149.887 ppl - 7 depth - 10277 nodes - 55586 customers
Epoch 24 / 100 - 3269.1 lps - 144.077 ppl - 7 depth - 10325 nodes - 56160 customers
Epoch 25 / 100 - 5762.8 lps - 143.742 ppl - 9 depth - 10371 nodes - 56348 customers
Epoch 26 / 100 - 5762.8 lps - 141.674 ppl - 7 depth - 10573 nodes - 56961 customers
Epoch 27 / 100 - 6421.4 lps - 136.252 ppl - 7 depth - 10622 nodes - 57387 customers
Epoch 28 / 100 - 5689.9 lps - 135.364 ppl - 7 depth - 10835 nodes - 57874 customers
Epoch 29 / 100 - 6115.6 lps - 133.813 ppl - 8 depth - 10863 nodes - 57822 customers
Epoch 30 / 100 - 6200.0 lps - 133.984 ppl - 8 depth - 11053 nodes - 58280 customers
Epoch 31 / 100 - 4781.9 lps - 130.355 ppl - 7 depth - 11375 nodes - 58662 customers
Epoch 32 / 100 - 5837.7 lps - 127.931 ppl - 7 depth - 11467 nodes - 58921 customers
Epoch 33 / 100 - 5319.5 lps - 129.356 ppl - 8 depth - 11507 nodes - 59012 customers
Epoch 34 / 100 - 6074.3 lps - 131.735 ppl - 7 depth - 11485 nodes - 59099 customers
Epoch 35 / 100 - 6074.3 lps - 125.482 ppl - 7 depth - 11747 nodes - 59608 customers
Epoch 36 / 100 - 5448.5 lps - 122.808 ppl - 7 depth - 11832 nodes - 60020 customers
Epoch 37 / 100 - 5837.7 lps - 119.554 ppl - 8 depth - 12008 nodes - 60393 customers
Epoch 38 / 100 - 5515.3 lps - 122.608 ppl - 7 depth - 11933 nodes - 60279 customers
Epoch 39 / 100 - 5583.9 lps - 119.074 ppl - 7 depth - 11952 nodes - 60321 customers
Epoch 40 / 100 - 6243.1 lps - 123.656 ppl - 7 depth - 11748 nodes - 60134 customers
Epoch 41 / 100 - 5618.8 lps - 118.181 ppl - 7 depth - 11871 nodes - 60465 customers
Epoch 42 / 100 - 5875.8 lps - 118.811 ppl - 7 depth - 12175 nodes - 60925 customers
Epoch 43 / 100 - 6286.7 lps - 116.837 ppl - 7 depth - 12175 nodes - 60941 customers
Epoch 44 / 100 - 5549.4 lps - 119.270 ppl - 8 depth - 11943 nodes - 60779 customers
Epoch 45 / 100 - 5837.7 lps - 115.230 ppl - 6 depth - 11760 nodes - 60592 customers
Epoch 46 / 100 - 5762.8 lps - 116.281 ppl - 8 depth - 11937 nodes - 60910 customers
Epoch 47 / 100 - 5800.0 lps - 115.869 ppl - 7 depth - 12034 nodes - 60900 customers
Epoch 48 / 100 - 6375.9 lps - 119.424 ppl - 7 depth - 11964 nodes - 60862 customers
Epoch 49 / 100 - 5618.8 lps - 121.917 ppl - 8 depth - 11973 nodes - 60896 customers
Epoch 50 / 100 - 5837.7 lps - 118.818 ppl - 7 depth - 12072 nodes - 61010 customers
Epoch 51 / 100 - 5549.4 lps - 119.322 ppl - 7 depth - 12293 nodes - 61346 customers
Epoch 52 / 100 - 5800.0 lps - 119.690 ppl - 9 depth - 12237 nodes - 61430 customers
Epoch 53 / 100 - 5993.3 lps - 116.886 ppl - 7 depth - 12380 nodes - 61721 customers
Epoch 54 / 100 - 5448.5 lps - 116.629 ppl - 7 depth - 12365 nodes - 61734 customers
Epoch 55 / 100 - 5800.0 lps - 114.435 ppl - 7 depth - 12653 nodes - 62373 customers
Epoch 56 / 100 - 5481.7 lps - 115.507 ppl - 7 depth - 12998 nodes - 62854 customers
Epoch 57 / 100 - 4966.9 lps - 117.921 ppl - 8 depth - 12803 nodes - 62594 customers
Epoch 58 / 100 - 6157.5 lps - 112.332 ppl - 7 depth - 12935 nodes - 62742 customers
Epoch 59 / 100 - 5481.7 lps - 113.768 ppl - 7 depth - 13057 nodes - 62888 customers
Epoch 60 / 100 - 5654.1 lps - 112.779 ppl - 7 depth - 12775 nodes - 62678 customers
Epoch 61 / 100 - 5415.7 lps - 112.597 ppl - 7 depth - 12820 nodes - 62942 customers
Epoch 62 / 100 - 6157.5 lps - 113.605 ppl - 8 depth - 12815 nodes - 63025 customers
Epoch 63 / 100 - 6375.9 lps - 112.919 ppl - 8 depth - 12789 nodes - 63262 customers
Epoch 64 / 100 - 5351.2 lps - 111.276 ppl - 8 depth - 12877 nodes - 63258 customers
Epoch 65 / 100 - 5415.7 lps - 110.288 ppl - 7 depth - 12884 nodes - 63592 customers
Epoch 66 / 100 - 5319.5 lps - 113.372 ppl - 7 depth - 12903 nodes - 63573 customers
Epoch 67 / 100 - 5875.8 lps - 114.626 ppl - 8 depth - 12864 nodes - 63687 customers
Epoch 68 / 100 - 5762.8 lps - 109.990 ppl - 8 depth - 13007 nodes - 63963 customers
Epoch 69 / 100 - 5654.1 lps - 108.115 ppl - 7 depth - 13112 nodes - 64197 customers
Epoch 70 / 100 - 4807.5 lps - 107.138 ppl - 8 depth - 13052 nodes - 63982 customers
Epoch 71 / 100 - 5481.7 lps - 107.296 ppl - 8 depth - 13312 nodes - 64516 customers
Epoch 72 / 100 - 5726.1 lps - 108.629 ppl - 7 depth - 13324 nodes - 64414 customers
Epoch 73 / 100 - 5288.2 lps - 107.786 ppl - 8 depth - 13320 nodes - 64468 customers
Epoch 74 / 100 - 5762.8 lps - 110.619 ppl - 9 depth - 13491 nodes - 64624 customers
Epoch 75 / 100 - 4994.4 lps - 106.170 ppl - 8 depth - 13441 nodes - 64546 customers
Epoch 76 / 100 - 5762.8 lps - 106.992 ppl - 9 depth - 13380 nodes - 64735 customers
Epoch 77 / 100 - 5762.8 lps - 106.842 ppl - 8 depth - 13379 nodes - 64687 customers
Epoch 78 / 100 - 5196.5 lps - 105.710 ppl - 7 depth - 13550 nodes - 64863 customers
Epoch 79 / 100 - 5515.3 lps - 106.121 ppl - 8 depth - 13373 nodes - 64649 customers
Epoch 80 / 100 - 5257.3 lps - 106.722 ppl - 7 depth - 13423 nodes - 64574 customers
Epoch 81 / 100 - 5726.1 lps - 108.408 ppl - 7 depth - 13527 nodes - 64629 customers
Epoch 82 / 100 - 5415.7 lps - 105.506 ppl - 8 depth - 13568 nodes - 64870 customers
Epoch 83 / 100 - 5549.4 lps - 107.099 ppl - 10 depth - 13524 nodes - 65009 customers
Epoch 84 / 100 - 5351.2 lps - 107.436 ppl - 8 depth - 13359 nodes - 64745 customers
Epoch 85 / 100 - 5415.7 lps - 107.840 ppl - 7 depth - 13240 nodes - 64688 customers
Epoch 86 / 100 - 5914.5 lps - 107.080 ppl - 8 depth - 13171 nodes - 64507 customers
Epoch 87 / 100 - 5351.2 lps - 108.628 ppl - 8 depth - 13260 nodes - 64503 customers
Epoch 88 / 100 - 5689.9 lps - 108.493 ppl - 8 depth - 13317 nodes - 64647 customers
Epoch 89 / 100 - 5383.2 lps - 109.974 ppl - 7 depth - 13217 nodes - 64434 customers
Epoch 90 / 100 - 5953.6 lps - 110.797 ppl - 7 depth - 13216 nodes - 64478 customers
Epoch 91 / 100 - 6074.3 lps - 110.236 ppl - 7 depth - 13289 nodes - 64671 customers
Epoch 92 / 100 - 5288.2 lps - 108.192 ppl - 9 depth - 13258 nodes - 64709 customers
Epoch 93 / 100 - 5689.9 lps - 109.370 ppl - 8 depth - 13435 nodes - 64840 customers
Epoch 94 / 100 - 5383.2 lps - 107.656 ppl - 7 depth - 13337 nodes - 64800 customers
Epoch 95 / 100 - 5914.5 lps - 108.081 ppl - 7 depth - 13388 nodes - 64869 customers
Epoch 96 / 100 - 5953.6 lps - 108.045 ppl - 7 depth - 13430 nodes - 64744 customers
Epoch 97 / 100 - 5319.5 lps - 110.287 ppl - 7 depth - 13245 nodes - 64377 customers
Epoch 98 / 100 - 5549.4 lps - 110.845 ppl - 8 depth - 13214 nodes - 64544 customers
Epoch 99 / 100 - 5288.2 lps - 107.337 ppl - 8 depth - 13161 nodes - 64625 customers
Epoch 100 / 100 - 6115.6 lps - 109.387 ppl - 8 depth - 13412 nodes - 64795 customers
8
13412
64795
35707
51639
文章を生成しています ...
" Perhaps it doesn't matter which was the first figure , the whole place , and made believe to worry 
" Hold your tongue hanging out of its mouth and the March Hare was I think I can say . " 
Soo  oop ! 
" I can't show you ! " said Alice . 
The King laid for some time without hearing anything more till the puppy's bark just as she had never heard it . ) 
" Of course , in a shrill , loud . 
And she began shrinking directly . " 
" Hold your tongue ! " and the words :  
" You might find another minute there seemed to be no chance of getting up and went on growing , and she felt very grave that she was now more the pig , my poor little thing sat down and began to cry again . " I quite a commotion in the last few yards off after a few minutes , and very soon came upon a little three - legged stool in the air . " If I don't know what to do that in a low , and their curls got thrown out of a tree , she was now about in a melancholy tone . And she's such a thing before the trial's over ! " 
I see what was the fan and gloves , and she tried to get out again :  
" And here , I think you'd better , or at any rate , " said Alice . 
Then the Dormouse again took the place on . Her chin was nothing . 
" It's a friend . She was walking hand in her life ! " shouted the Queen . " We know it to be lost : away the time she went on , " said Alice . 
But , it would have this cat removed , " said Alice . 
" Well , if I fell off , and found herself safe in a thick wood . 
" Not yet what a Mock Turtle . 
" Once upon a low voice , and all the other , trying to find that she began shrinking directly . 
The judge , I'll stay down on their slates , and that you couldn't answer to by the pope 
" Well , at the top of the Mock Turtle . " 
When the procession , wondering how she would manage to do it again : they would call it : " and then said , turning to Alice , a good deal : you'd take a fancy , that : he kept all over their slates , " She said it to the Dormouse is asleep again , and Alice was very nearly carried the pepper that it might be rude 
The next thing , and the Queen . " And what are they had a head could be no use their eyes appeared , and then said , " It was the best butter in the pool a little of the hall . " 
How doth the little door , and Alice was very uncomfortable . 
" I can't tell you my mind about like that ! " 
" What for ?  " 
" I can't be , " Alice replied very readily : wouldn't talk , " said Alice . 
This answer questions , first form into the garden . " I'm glad I had not gone 
" It's a friend . 
" Who are you ? " said the Hatter . 
" I'm not a very difficult question , " continued the Gryphon , and the Queen , the Duchess's cook . 
" No , no ! Youre thinking about her pet : she thought at first , " and the other , trying to touch her as she spoke . 
Soup of the evening , beautiful Soup ! " cried the Mouse , who seemed to be no chance of this . " Oh , my dear : she found herself falling through the door , and went on : " But what happens . " Do you mean by this time , " she said to herself , " I should think very little use , as the soldiers , or I'll have you executed on the door , and was coming . It was a dead silence . " 
" That would be quite as much as serpents night , I beg your pardon , and he wasn't much , " said Alice . 
" Come , let's try Geography 
So she set to work throwing everything seemed to her . 
And argued each side of what ? " said the Hatter . 
" What a pity it can't be said . 
" Well , perhaps you were trying  " 
" If you can't be a book . The poor little thing sat down and began to cry again . 
" No , " she said to herself , " I thought it would be the right size : why , if I fell off to the beginning to get very tired of this sort of idea that they couldn't have done , " she said to herself , " I haven't the least  at least , " I can't be so proud as all the jurymen . 
Alice was not a very difficult question , and I don't like them about it , you know , " said Alice . 
" I can't show you our heads downward ! " said Alice . 
" And so she went on , " ( Alice had no idea what to do it ! Oh , you know , " said Alice . 
This answer , so she went on , " said Alice . 
" It isn't usual , " said Alice . 
" Exactly so , very good opportunity for repeating " You might as well wait , it was , even if I shall fall a long time with great emphasis , looking anxiously among the trees behind it , " said Alice . 
And she thought it over a child , " said Alice . 
A large she had peeped out at the door of the Rabbit's voice , and the Queen , the Owl had grown woman  how is it ? " Alice asked in a low , and their curls got to ? " 
" Never ! " said the Hatter . 
" I can't be a little nervous or not . 
深さ6の句を表示します ...
steam - engine when she caught 
promised to tell me your history 
she went on all the same 
and looking at the Cat's head 
what was going to happen next 
) could not make out at 
of long ago : and how 
( luckily the salt water had 
puppy made another rush at the 
she came in sight of the 
" said the Cat , and 
one in by mistake ; and 
had asked it aloud ; and 
housemaid , " she said to 
of the Lobster Quadrille , that 
are you fond  of  
the moral of that is  
said the Dormouse : " not 
the judge , " she said 
! " shouted the Queen . 
, then , ' the Gryphon 
out of the way  " 
not used to it ! " 
" Anything you like , " 
Of course it is , " 
was said to live . " 
grass , merely remarking as it 
of little birds and beasts , 
back to the Cheshire Cat , 
, and was going off into 
as they came nearer , Alice 
master says youre to go down
```

