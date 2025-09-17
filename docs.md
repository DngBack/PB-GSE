1) Motivation & Problem

Bài toán. Phân loại với quyền từ chối (abstention) trên dữ liệu long-tail. Chia các lớp vào 
𝐾
K nhóm 
𝐺
1
,
…
,
𝐺
𝐾
G
1
	​

,…,G
K
	​

 (ví dụ head/tail). Với cặp bộ quy tắc 
(
ℎ
,
𝑟
)
(h,r) (classifier 
ℎ
:
𝑋
→
𝑌
h:X→Y, rejector 
𝑟
:
𝑋
→
{
0
,
1
}
r:X→{0,1} với 
𝑟
=
1
r=1 là reject), balanced selective risk là

	
𝑅
bal
rej
(
ℎ
,
𝑟
)
  
=
  
1
𝐾
∑
𝑘
=
1
𝐾
Pr
⁡
 ⁣
(
𝑦
≠
ℎ
(
𝑥
)
∣
𝑟
(
𝑥
)
=
0
,
  
𝑦
∈
𝐺
𝑘
)
  
+
  
𝑐
 
Pr
⁡
(
𝑟
(
𝑥
)
=
1
)
.
		
(1)
R
bal
rej
	​

(h,r)=
K
1
	​

k=1
∑
K
	​

Pr(y

=h(x)∣r(x)=0,y∈G
k
	​

)+cPr(r(x)=1).
(1)

Khó khăn. (i) Dữ liệu lệch làm posterior ước lượng ở tail kém tin cậy; (ii) rejector “chuẩn” kiểu Chow (ngưỡng 
1
−
𝑐
1−c) thiên vị head; (iii) nếu thiết kế gating theo mẫu để trộn nhiều mô hình (ensemble) thì rất dễ overfit tail.

Ý tưởng PB-GSE.

Dùng ensemble 
𝑓
1
,
…
,
𝑓
𝑀
f
1
	​

,…,f
M
	​

 (đã huấn luyện & hiệu chỉnh), với phân phối dự đoán 
𝑝
(
𝑚
)
(
𝑥
)
∈
Δ
𝐶
−
1
p
(m)
(x)∈Δ
C−1
.

Học gating theo mẫu 
𝑤
(
𝑥
;
𝜃
)
∈
Δ
𝑀
−
1
w(x;θ)∈Δ
M−1
 tạo ra

	
𝑝
𝑄
𝜃
(
𝑥
)
  
=
  
∑
𝑚
=
1
𝑀
𝑤
𝑚
(
𝑥
;
𝜃
)
 
𝑝
(
𝑚
)
(
𝑥
)
.
		
(2)
p
Q
θ
	​

	​

(x)=
m=1
∑
M
	​

w
m
	​

(x;θ)p
(m)
(x).
(2)

Suy luận abstention theo cân bằng nhóm bằng quy tắc Bayes của bài toán tuyến tính (bên dưới) cắm 
𝑝
𝑄
𝜃
p
Q
θ
	​

	​

 vào (thay 
𝜂
η).

Huấn luyện 
𝜃
θ không tối ưu loss trực tiếp mà tối ưu một bound PAC-Bayes cho (1), nhằm khống chế overfit (đặc biệt ở tail).

2) Contributions (viết thẳng vào abstract/introduction)

Bound PAC-Bayes cho balanced selective risk. Chúng tôi phát biểu và chứng minh một tổng quát hoá PAC-Bayes cho rủi ro (1) thông qua một biểu thức tuyến tính tương đương có ràng buộc, cho phép áp dụng trực tiếp vào mô hình có reject và chia nhóm, không cần surrogate.

Gating-by-Bound. Đề xuất thuật toán học gating 
𝑤
(
𝑥
;
𝜃
)
w(x;θ) bằng cách tối thiểu hoá cận trên PAC-Bayes của balanced selective risk; trong vòng trong dùng oracle rule (Theorem 1 trong công trình gốc) với 
𝑝
𝑄
𝜃
p
Q
θ
	​

	​

.

Group-aware prior/budget. Thiết kế prior 
Π
Π (hoặc ngân sách KL) phụ thuộc nhóm (head/tail), giúp đẩy năng lực ensemble về tail một cách có nguyên lý.

Bảo đảm lý thuyết. (i) Plug-in optimality: với 
𝛼
,
𝜇
α,μ cố định, quy tắc chọn lớp/abstain tối ưu trên lớp rule dựa trên 
𝑝
𝑄
𝜃
p
Q
θ
	​

	​

 có dạng đóng (cùng cấu trúc với Theorem 1); (ii) PAC-Bayes bound với loss bị chặn; (iii) mở rộng worst-group với no-regret EG (ăn khớp Theorem 3 “mixture of abstainers”).

3) Method
3.1. Tuyến tính hoá rủi ro cân bằng (dùng biến phụ)

Đặt

	
𝛼
𝑘
  
≜
  
𝐾
 
Pr
⁡
(
𝑟
(
𝑥
)
=
0
,
  
𝑦
∈
𝐺
𝑘
)
,
𝛼
𝑘
∈
(
0
,
1
]
.
		
(3)
α
k
	​

≜KPr(r(x)=0,y∈G
k
	​

),α
k
	​

∈(0,1].
(3)

Khi đó (chuẩn hoá như trong công trình gốc), rủi ro cân bằng tương đương với mục tiêu tuyến tính có ràng buộc:

	
∑
𝑘
=
1
𝐾
1
𝛼
𝑘
Pr
⁡
(
𝑦
≠
ℎ
(
𝑥
)
,
 
𝑟
(
𝑥
)
=
0
,
 
𝑦
∈
𝐺
𝑘
)
  
+
  
𝑐
 
Pr
⁡
(
𝑟
(
𝑥
)
=
1
)
⏟
𝑅
𝛼
lin
(
ℎ
,
𝑟
)
s.t.
𝛼
𝑘
  
=
  
𝐾
Pr
⁡
(
𝑟
=
0
,
𝑦
∈
𝐺
𝑘
)
.
		
(4)
R
α
lin
	​

(h,r)
k=1
∑
K
	​

α
k
	​

1
	​

Pr(y

=h(x),r(x)=0,y∈G
k
	​

)+cPr(r(x)=1)
	​

	​

s.t.α
k
	​

=KPr(r=0,y∈G
k
	​

).
(4)

(4) là đúng theo phép tách mẫu số/biến phụ tiêu chuẩn; nghiệm thoả KKT cho 
𝛼
𝑘
α
k
	​

 khôi phục (1). Ta sẽ dùng (4) để vừa suy luận Bayes, vừa “đưa loss về dạng bị chặn” cho PAC-Bayes.

3.2. Quy tắc Bayes plug-in trên 
𝑝
𝑄
𝜃
p
Q
θ
	​

	​


Với 
𝜂
(
⋅
)
η(⋅) chưa biết, ta plug-in 
𝑝
𝑄
𝜃
(
𝑥
)
p
Q
θ
	​

	​

(x) (2) vào quy tắc Bayes tối ưu của (4). Cụ thể, đặt chỉ số nhóm 
[
𝑦
]
[y] cho lớp 
𝑦
y và các bội số Lagrange 
𝜇
𝑘
μ
k
	​

, ta có:

Định lý 1 (Plug-in optimality trên 
𝑝
𝑄
𝜃
p
Q
θ
	​

	​

).
Cho 
𝛼
∈
(
0
,
1
]
𝐾
α∈(0,1]
K
 và 
𝜇
∈
𝑅
𝐾
μ∈R
K
 cố định. Xét cặp rule 
(
ℎ
𝜃
,
𝑟
𝜃
)
(h
θ
	​

,r
θ
	​

) xác định bởi

	
ℎ
𝜃
(
𝑥
)
  
=
  
arg
⁡
max
⁡
𝑦
  
𝑝
𝑄
𝜃
,
𝑦
(
𝑥
)
𝛼
[
𝑦
]
,
		
(5a)
h
θ
	​

(x)=arg
y
max
	​

α
[y]
	​

p
Q
θ
	​

,y
	​

(x)
	​

,
(5a)
	
𝑟
𝜃
(
𝑥
)
=
1
 
⟺
 
max
⁡
𝑦
𝑝
𝑄
𝜃
,
𝑦
(
𝑥
)
𝛼
[
𝑦
]
  
<
  
∑
𝑗
 ⁣
(
1
𝛼
[
𝑗
]
−
𝜇
[
𝑗
]
)
𝑝
𝑄
𝜃
,
𝑗
(
𝑥
)
  
−
  
𝑐
.
		
(5b)
r
θ
	​

(x)=1 ⟺ 
y
max
	​

α
[y]
	​

p
Q
θ
	​

,y
	​

(x)
	​

<
j
∑
	​

(
α
[j]
	​

1
	​

−μ
[j]
	​

)p
Q
θ
	​

,j
	​

(x)−c.
(5b)

Khi đó, trong lớp các rule chỉ phụ thuộc vào 
𝑝
𝑄
𝜃
p
Q
θ
	​

	​

, 
(
ℎ
𝜃
,
𝑟
𝜃
)
(h
θ
	​

,r
θ
	​

) tối ưu 
𝑅
𝛼
lin
R
α
lin
	​

 ở (4).
Phác hoạ chứng minh. Lập Lagrangian của (4) theo cách chuẩn; tối ưu cục bộ theo từng 
𝑥
x dẫn đến chọn lớp bằng 
arg
⁡
max
⁡
𝑦
𝑝
𝑄
𝜃
,
𝑦
/
𝛼
[
𝑦
]
argmax
y
	​

p
Q
θ
	​

,y
	​

/α
[y]
	​

 và so sánh chấp nhận vs reject bằng ngưỡng tuyến tính trên toàn vector 
𝑝
𝑄
𝜃
(
𝑥
)
p
Q
θ
	​

	​

(x) (như (5b)). Điểm mấu chốt: dạng tuyến tính theo 
𝑝
p bảo toàn dưới phép thay 
𝜂
↦
𝑝
𝑄
𝜃
η↦p
Q
θ
	​

	​

.

Ghi chú. Khi 
𝛼
,
𝜇
α,μ được thay bởi nghiệm cố định-điểm (ước lượng từ tập hiệu chỉnh), (5) chính là quy tắc suy luận của PB-GSE.

3.3. Huấn luyện 
𝑤
(
𝑥
;
𝜃
)
w(x;θ) bằng PAC-Bayes

Ta xem 
𝜃
θ là tham số của gating; xác suất hiệp biến của 
𝜃
θ được mô hình hoá bởi posterior 
𝑄
Q (ví dụ Gaussian) và prior 
Π
Π (Gaussian chuẩn, hay prior nhóm-nhạy). Với 
𝜃
∼
𝑄
θ∼Q, predictor là rule 
(
ℎ
𝜃
,
𝑟
𝜃
)
(h
θ
	​

,r
θ
	​

) theo (5). Định nghĩa loss bị chặn cho (4):

	
ℓ
𝛼
(
𝜃
;
(
𝑥
,
𝑦
)
)
  
=
  
∑
𝑘
=
1
𝐾
1
𝛼
𝑘
 
1
[
𝑦
∈
𝐺
𝑘
]
  
1
[
𝑟
𝜃
(
𝑥
)
=
0
]
  
1
[
ℎ
𝜃
(
𝑥
)
≠
𝑦
]
  
+
  
𝑐
 
1
[
𝑟
𝜃
(
𝑥
)
=
1
]
.
		
(6)
ℓ
α
	​

(θ;(x,y))=
k=1
∑
K
	​

α
k
	​

1
	​

1[y∈G
k
	​

]1[r
θ
	​

(x)=0]1[h
θ
	​

(x)

=y]+c1[r
θ
	​

(x)=1].
(6)

Rõ ràng 
0
≤
ℓ
𝛼
≤
𝐿
𝛼
:
=
max
⁡
{
𝑐
,
 
∑
𝑘
1
𝛼
𝑘
}
+
𝑐
0≤ℓ
α
	​

≤L
α
	​

:=max{c,∑
k
	​

α
k
	​

1
	​

}+c. Ta chuẩn hoá 
ℓ
~
𝛼
=
ℓ
𝛼
/
𝐿
𝛼
∈
[
0
,
1
]
ℓ
~
α
	​

=ℓ
α
	​

/L
α
	​

∈[0,1].

4) Theory
4.1. PAC-Bayes bound cho balanced selective risk tuyến tính

Định lý 2 (PAC-Bayes cho 
ℓ
~
𝛼
ℓ
~
α
	​

).
Cho mẫu i.i.d. 
𝑆
=
{
(
𝑥
𝑖
,
𝑦
𝑖
)
}
𝑖
=
1
𝑛
∼
𝐷
𝑛
S={(x
i
	​

,y
i
	​

)}
i=1
n
	​

∼D
n
. Với mọi prior 
Π
Π trên 
𝜃
θ (độc lập 
𝑆
S), với mọi posterior 
𝑄
Q (có thể phụ thuộc 
𝑆
S), với mọi 
𝛿
∈
(
0
,
1
)
δ∈(0,1), với xác suất 
≥
1
−
𝛿
≥1−δ trên 
𝑆
S, ta có

	
𝐸
𝜃
∼
𝑄
[
𝑅
𝛼
lin
(
ℎ
𝜃
,
𝑟
𝜃
)
]
  
≤
  
𝐸
𝜃
∼
𝑄
[
𝑅
^
𝛼
,
𝑆
lin
(
ℎ
𝜃
,
𝑟
𝜃
)
]
  
+
  
𝐿
𝛼
K
L
(
𝑄
∥
Π
)
+
ln
⁡
 ⁣
2
𝑛
𝛿
2
(
𝑛
−
1
)
.
		
(7)
E
θ∼Q
	​

[R
α
lin
	​

(h
θ
	​

,r
θ
	​

)]≤E
θ∼Q
	​

[
R
α,S
lin
	​

(h
θ
	​

,r
θ
	​

)]+L
α
	​

2(n−1)
KL(Q∥Π)+ln
δ
2
n
	​

	​

	​

	​

.
(7)

Trong đó 
𝑅
^
𝛼
,
𝑆
lin
R
α,S
lin
	​

 là trung bình mẫu của 
ℓ
𝛼
ℓ
α
	​

 (không chuẩn hoá).
Phác hoạ chứng minh. Áp dụng PAC-Bayes cổ điển (Seeger/Catoni) cho loss bị chặn 
ℓ
~
𝛼
∈
[
0
,
1
]
ℓ
~
α
	​

∈[0,1], rồi nhân lại bởi 
𝐿
𝛼
L
α
	​

. Quy tắc 
(
ℎ
𝜃
,
𝑟
𝜃
)
(h
θ
	​

,r
θ
	​

) là xác định với 
𝜃
θ, nên toàn bộ ngẫu nhiên nằm ở 
𝑄
Q.

Bình luận. (7) là cận trên rủi ro tuyến tính 
𝑅
𝛼
lin
R
α
lin
	​

. Khi 
𝛼
α chọn đúng nghiệm KKT (hoặc được ước lượng nhất quán), chênh lệch giữa 
𝑅
𝛼
lin
R
α
lin
	​

 và balanced risk (1) là bù trừ nhỏ (mục 4.3). Vì vậy tối thiểu hoá vế phải (7) là hợp lý cho balanced selective risk.

4.2. Group-aware prior/budget

Để ưu tiên tail, có hai đường đúng nguyên lý:

Prior nhóm-nhạy. Đặt 
Π
=
𝑁
(
0
,
𝜎
2
𝐼
)
Π=N(0,σ
2
I) nhưng chuẩn hoá target của gating sao cho khi 
𝑥
x thuộc vùng tail (phát hiện bằng meta-đặc trưng hoặc nhãn nhóm), mạng gating có độ tự do lớn hơn (ví dụ thêm feat vào 
𝑤
w), đồng thời giữ 
𝜎
σ lớn hơn — làm KL(Q|\Pi) nhỏ hơn khi 
𝑄
Q dịch chuyển để phục vụ tail.

KL-budget theo nhóm. Tối ưu (7) với ràng buộc 
𝐸
(
𝑥
,
𝑦
)
∈
𝑆
∩
𝐺
𝑘
K
L
(
𝑄
∥
Π
)
≤
𝐵
𝑘
E
(x,y)∈S∩G
k
	​

	​

KL(Q∥Π)≤B
k
	​

, với 
𝐵
tail
>
𝐵
head
B
tail
	​

>B
head
	​

. Điều này cho năng lực biểu diễn gating lớn hơn ở tail nhưng vẫn trong biên PAC-Bayes.

4.3. Từ rủi ro tuyến tính về balanced risk

Khi 
𝛼
α là nghiệm KKT 
𝛼
⋆
=
𝐾
Pr
⁡
(
𝑟
𝜃
=
0
,
𝑦
∈
𝐺
𝑘
)
α
⋆
=KPr(r
θ
	​

=0,y∈G
k
	​

), ta có 
𝑅
𝛼
⋆
lin
(
ℎ
𝜃
,
𝑟
𝜃
)
=
𝑅
bal
rej
(
ℎ
𝜃
,
𝑟
𝜃
)
R
α
⋆
lin
	​

(h
θ
	​

,r
θ
	​

)=R
bal
rej
	​

(h
θ
	​

,r
θ
	​

). Trong thực hành, 
𝛼
α ước lượng từ một tập hiệu chỉnh độc lập 
𝑆
cal
S
cal
	​

 bằng cố định-điểm:

	
𝛼
^
𝑘
  
=
  
𝐾
 
Pr
⁡
𝑆
cal
 ⁣
(
𝑟
𝜃
(
𝑥
)
=
0
,
  
𝑦
∈
𝐺
𝑘
)
.
		
(8)
α
^
k
	​

=K
S
cal
	​

Pr
	​

(r
θ
	​

(x)=0,y∈G
k
	​

).
(8)

Giả sử 
𝛼
^
→
𝛼
⋆
α
^
→α
⋆
 theo xác suất khi 
∣
𝑆
cal
∣
→
∞
∣S
cal
	​

∣→∞ (điều kiện đều hoá nhẹ). Khi đó

	
∣
𝑅
bal
rej
(
ℎ
𝜃
,
𝑟
𝜃
)
−
𝑅
𝛼
^
lin
(
ℎ
𝜃
,
𝑟
𝜃
)
∣
  
≤
  
∑
𝑘
=
1
𝐾
∣
1
𝛼
𝑘
⋆
−
1
𝛼
^
𝑘
∣
 
Pr
⁡
(
𝑦
≠
ℎ
𝜃
,
𝑟
𝜃
=
0
,
𝑦
∈
𝐺
𝑘
)
  
→
𝑝
  
0.
		
(9)
	​

R
bal
rej
	​

(h
θ
	​

,r
θ
	​

)−R
α
^
lin
	​

(h
θ
	​

,r
θ
	​

)
	​

≤
k=1
∑
K
	​

	​

α
k
⋆
	​

1
	​

−
α
^
k
	​

1
	​

	​

Pr(y

=h
θ
	​

,r
θ
	​

=0,y∈G
k
	​

)
p
	​

0.
(9)

Vì vậy, tối ưu bound (7) với 
𝛼
=
𝛼
^
α=
α
^
 là nhất quán cho balanced risk.

4.4. Worst-group extension (mixture of abstainers)

Cho 
𝜓
(
𝑒
1
,
…
,
𝑒
𝐾
)
=
max
⁡
𝑘
𝑒
𝑘
ψ(e
1
	​

,…,e
K
	​

)=max
k
	​

e
k
	​

. Đặt trọng số nhóm 
𝛽
∈
Δ
𝐾
−
1
β∈Δ
K−1
. Với mỗi 
𝛽
β, ta tối ưu rủi ro 
∑
𝑘
𝛽
𝑘
𝑒
𝑘
∑
k
	​

β
k
	​

e
k
	​

 bằng (5)–(7) (thay loss theo 
𝛽
β). Vòng ngoài cập nhật 
𝛽
β bằng Exponentiated-Gradient (EG) (no-regret). Kết quả chuẩn: sau 
𝑇
T vòng, nghiệm ngẫu nhiên (hoặc trung bình) đạt sai số tối ưu 
𝑂
(
log
⁡
𝐾
/
𝑇
)
O(
logK/T
	​

) cộng các sai số ước lượng trong vòng trong (PAC-Bayes).

5) Thuật toán (tối thiểu bound)

Input. Base posteriors 
𝑝
(
𝑚
)
(
𝑥
)
p
(m)
(x), tập train 
𝑆
tr
S
tr
	​

, tập hiệu chỉnh 
𝑆
cal
S
cal
	​

, tập đánh giá 
𝑆
S (độc lập cho bound), chi phí reject 
𝑐
c.

Bước 0. Huấn luyện/cân bằng base models; calibrate (temperature / vector scaling) để 
𝑝
(
𝑚
)
p
(m)
 tin cậy theo nhóm.

Bước 1 (Khởi tạo). Chọn prior 
Π
Π (Gaussian 
𝑁
(
0
,
𝜎
2
𝐼
)
N(0,σ
2
I) hoặc prior nhóm-nhạy). Khởi tạo posterior 
𝑄
=
𝑁
(
𝜃
0
,
𝑠
0
2
𝐼
)
Q=N(θ
0
	​

,s
0
2
	​

I).

Bước 2 (Cố định-điểm 
𝛼
α). Với 
𝜃
∼
𝑄
θ∼Q (lấy mẫu hoặc dùng trung bình), tính 
𝑝
𝑄
𝜃
p
Q
θ
	​

	​

 (2); suy ra 
(
ℎ
𝜃
,
𝑟
𝜃
)
(h
θ
	​

,r
θ
	​

) theo (5); cập nhật 
𝛼
^
α
^
 theo (8) trên 
𝑆
cal
S
cal
	​

. Lặp đến hội tụ (10–20 vòng; nhanh).

Bước 3 (Minimise PAC-Bayes upper-bound). Trên tập 
𝑆
S (độc lập 
𝑆
cal
S
cal
	​

), tối thiểu hoá

	
𝑈
(
𝑄
)
  
=
  
𝐸
𝜃
∼
𝑄
[
𝑅
^
𝛼
^
,
𝑆
lin
(
ℎ
𝜃
,
𝑟
𝜃
)
]
  
+
  
𝐿
𝛼
^
K
L
(
𝑄
∥
Π
)
+
ln
⁡
 ⁣
2
𝑛
𝛿
2
(
𝑛
−
1
)
  
+
  
𝜆
 
Ω
fair
(
𝑄
)
,
		
(10)
U(Q)=E
θ∼Q
	​

[
R
α
^
,S
lin
	​

(h
θ
	​

,r
θ
	​

)]+L
α
^
	​

2(n−1)
KL(Q∥Π)+ln
δ
2
n
	​

	​

	​

	​

+λΩ
fair
	​

(Q),
(10)

với 
Ω
fair
Ω
fair
	​

 (tuỳ chọn) phạt lệch accept-rate giữa nhóm. Thực tế, ta dùng reparameterization (Gaussian 
𝑄
Q) để ước lượng gradient Monte-Carlo của vế trái (10); 
K
L
KL có dạng đóng.

Bước 4 (Inference). Dùng 
𝑄
Q cuối (hoặc trung bình) để xác định 
𝑤
(
𝑥
;
𝜃
)
w(x;θ), tính 
𝑝
𝑄
𝜃
(
𝑥
)
p
Q
θ
	​

	​

(x), áp dụng (5) với 
𝛼
^
,
𝜇
^
α
^
,
μ
^
	​

 (từ vòng cố định-điểm) → dự đoán/abstain.

Worst-group (tuỳ chọn). Vòng ngoài cập nhật 
𝛽
β (EG), mỗi lần gọi Bước 2–3 với loss theo 
𝛽
β.

6) Bảo đảm đúng đắn & các lưu ý thực hành

Loss bị chặn. (6) bị chặn hữu hạn 
⇒
⇒ áp dụng PAC-Bayes chuẩn.

Độc lập mẫu cho bound. Để (7) “sạch”: dùng data-split—
𝛼
^
α
^
 học trên 
𝑆
cal
S
cal
	​

; bound (7) đánh giá trên 
𝑆
S độc lập. Khi cần one-split, dùng kỹ thuật ghost sample hoặc cross-fitting (trình bày trong phụ lục).

Gating là xác định theo 
𝜃
θ. PAC-Bayes xem ngẫu nhiên ở 
𝑄
Q (không ở 
𝑥
x) → hoàn toàn phù hợp: mọi phụ thuộc 
𝑥
x nằm trong hàm 
𝑤
(
𝑥
;
𝜃
)
w(x;θ) xác định.

Hiệu chỉnh xác suất. Vì (5) là tuyến tính theo 
𝑝
p, calibration theo nhóm (temperature/vector scaling) trước khi học gating giúp giảm nhiễu ngưỡng.

Tối ưu hoá. Trong thực nghiệm, có thể dùng Dirac-posterior (deterministic 
𝜃
θ) + regularizer KL surrogate (weight decay nhóm-nhạy) để gần với (10) khi compute hạn chế; bound chính thức yêu cầu 
𝑄
Q thực.

7) Phác thảo chứng minh (tập trung các điểm dễ sai)

(i) Tương đương (1)↔(4). Viết 
Pr
⁡
(
𝑦
≠
ℎ
∣
𝑟
=
0
,
𝑦
∈
𝐺
𝑘
)
=
Pr
⁡
(
𝑦
≠
ℎ
,
𝑟
=
0
,
𝑦
∈
𝐺
𝑘
)
/
Pr
⁡
(
𝑟
=
0
,
𝑦
∈
𝐺
𝑘
)
Pr(y

=h∣r=0,y∈G
k
	​

)=Pr(y

=h,r=0,y∈G
k
	​

)/Pr(r=0,y∈G
k
	​

). Đặt 
𝛼
𝑘
=
𝐾
Pr
⁡
(
𝑟
=
0
,
𝑦
∈
𝐺
𝑘
)
α
k
	​

=KPr(r=0,y∈G
k
	​

) 
⇒
⇒ (4). Nghiệm KKT khôi phục (1) tại 
𝛼
⋆
α
⋆
.

(ii) Plug-in optimality (Định lý 1). Lagrangian của (4):

𝐿
=
∑
𝑘
1
𝛼
𝑘
Pr
⁡
(
𝑦
≠
ℎ
,
𝑟
=
0
,
𝑦
∈
𝐺
𝑘
)
+
𝑐
Pr
⁡
(
𝑟
=
1
)
+
∑
𝑘
𝜇
𝑘
(
𝐾
Pr
⁡
(
𝑟
=
0
,
𝑦
∈
𝐺
𝑘
)
−
𝛼
𝑘
)
.
L=
k
∑
	​

α
k
	​

1
	​

Pr(y

=h,r=0,y∈G
k
	​

)+cPr(r=1)+
k
∑
	​

μ
k
	​

(KPr(r=0,y∈G
k
	​

)−α
k
	​

).

Phân rã theo 
𝑥
x và thay 
𝜂
η bởi 
𝑝
𝑄
𝜃
p
Q
θ
	​

	​

 cho ta hàm mục tiêu tuyến tính theo 
𝑝
𝑄
𝜃
(
𝑥
)
p
Q
θ
	​

	​

(x); tối ưu cục bộ sinh (5a)(5b).

(iii) PAC-Bayes (Định lý 2). Áp dụng bất đẳng thức PAC-Bayes cho loss bị chặn 
ℓ
~
𝛼
∈
[
0
,
1
]
ℓ
~
α
	​

∈[0,1]:

𝐸
𝑄
𝑅
≤
𝐸
𝑄
𝑅
^
+
(
K
L
(
𝑄
∥
Π
)
+
ln
⁡
(
2
𝑛
/
𝛿
)
)
/
2
(
𝑛
−
1
)
E
Q
	​

R≤E
Q
	​

R
^
+
(KL(Q∥Π)+ln(2
n
	​

/δ))/2(n−1)
	​

, sau đó nhân ngược 
𝐿
𝛼
L
α
	​

.

(iv) Nhất quán (9). Dưới điều kiện đều hoá nhẹ (ví dụ 
inf
⁡
𝑘
𝛼
𝑘
⋆
>
𝜖
>
0
inf
k
	​

α
k
⋆
	​

>ϵ>0), luật số lớn cho 
𝛼
^
α
^
 ⇒ 
𝛼
^
→
𝛼
⋆
α
^
→α
⋆
, và vì 
𝑥
↦
1
/
𝛼
x↦1/α liên tục trên 
[
𝜖
,
1
]
[ϵ,1], chênh lệch co lại về 0.

(v) Worst-group. EG là no-regret; khi vòng trong trả về lời giải xấp xỉ tối ưu theo 
𝛽
β, nghiệm hỗn hợp (trung bình theo vòng lặp) xấp xỉ chuẩn Bayes hỗn hợp (Theorem 3 kiểu mixture-of-abstainers).

8) Khuôn thực nghiệm (tối thiểu để publish)

Datasets: CIFAR-LT (imbalance 100/200), ImageNet-LT, iNaturalist-LT.

Baselines: CE+Chow; Balanced-CE+Chow; Bayes-plug-in (paper gốc); Deep Ensemble + Chow; Conformal selective (không LT).

PB-GSE configs:

Base: cRT, LDAM-DRW, CB-Focal (đa dạng kiến trúc/sampling).

Calibration per-group.

Prior 
Π
Π: Gaussian; phiên bản group-aware (σ lớn hơn cho tail).

Posterior 
𝑄
Q: Gaussian (diagonal) trên 
𝜃
θ của gating (MLP nhỏ).

Cố định-điểm 
𝛼
^
α
^
 trên 
𝑆
cal
S
cal
	​

.

Tối thiểu (10) trên 
𝑆
S (độc lập), chọn 
𝛿
=
0.05
δ=0.05.

Metrics: balanced selective error @coverage, worst-group selective error, AURC, accept-rate@group, ECE@group.

Ablations: vai trò prior nhóm-nhạy, cỡ 
𝑄
Q (variance), ảnh hưởng calibration, có/không fair regularizer, outer EG.

TL;DR cách triển khai an toàn

Huấn luyện M base models chú trọng tail; calibrate theo nhóm.

Học gating 
𝑤
(
𝑥
;
𝜃
)
w(x;θ) bằng cách tối thiểu hoá PAC-Bayes upper-bound (7)/(10) cho rủi ro tuyến tính (4); 
𝛼
α ước lượng bằng cố định-điểm trên tập hiệu chỉnh độc lập.

Suy luận bằng quy tắc (5) (cùng cấu trúc Bayes-optimal của công trình gốc), có thể mở rộng worst-group bằng EG.

nếu bạn muốn, mình sẽ chuyển toàn bộ phần “Theory” thành LaTeX (định lý, chứng minh, bổ đề & giả thiết đầy đủ) + pseudo-code cho vòng tối ưu (10) kèm cấu hình 
𝑄
,
Π
Q,Π (Gaussian) để bạn thả thẳng vào bản thảo.

You said:
Tôi muốn hướng dẫn triển khai implement thực nghiệm chi tiết cụ thể, để có thể viết thành paper. Cụ thể từ dữ liệu, xử lý dữ liệu, đánh giá metrics, .. 
ChatGPT said:
Thought for 1m 44s

Tuyệt—dưới đây là “playbook triển khai” cho PB-GSE: PAC-Bayes Group-Selective Ensemble đủ chi tiết để bạn code, chạy thí nghiệm và viết paper. Mình chia theo: dữ liệu → tiền xử lý → huấn luyện base models (long-tail aware) → hiệu chỉnh (calibration) → huấn luyện gating theo bound PAC-Bayes → suy luận theo quy tắc Theorem-1 plug-in với cố định-điểm 
𝛼
,
𝜇
α,μ → mở rộng worst-group → đánh giá & báo cáo. Toàn bộ đều độc lập công cụ (PyTorch/Lightning/Hydra đều áp được).

0) Tổng quan pipeline

Chuẩn bị dataset LT + chia nhóm (head/tail/…); tách train/val/cal/test.

Huấn luyện M base models (đa dạng loss/sampling/augment) + calibration theo nhóm → lưu xác suất 
𝑝
(
𝑚
)
(
𝑥
)
p
(m)
(x) cho {cal, val, test}.

Huấn luyện gating 
𝑤
(
𝑥
;
𝜃
)
w(x;θ) để tạo 
𝑝
𝑄
𝜃
(
𝑥
)
=
∑
𝑚
𝑤
𝑚
(
𝑥
;
𝜃
)
𝑝
(
𝑚
)
(
𝑥
)
p
Q
θ
	​

	​

(x)=∑
m
	​

w
m
	​

(x;θ)p
(m)
(x) bằng cách tối thiểu hoá cận PAC-Bayes cho rủi ro tuyến tính cân bằng (không tối ưu loss trực tiếp).

Tìm 
𝛼
^
,
𝜇
^
α
^
,
μ
^
	​

 bằng cố định-điểm trên tập cal (độc lập) theo quy tắc plug-in (Theorem-1).

Suy luận trên val/test: áp quy tắc plug-in (Theorem-1) với 
𝑝
𝑄
𝜃
p
Q
θ
	​

	​

, 
𝛼
^
,
𝜇
^
α
^
,
μ
^
	​

; nếu làm worst-group thì thêm vòng EG bên ngoài.

Đánh giá: balanced selective error / worst-group selective error, AURC, coverage & accept-rate@group, ECE@group; ablation & phân tích.

1) Dữ liệu & chia nhóm
1.1 Datasets khuyến nghị

CIFAR-10-LT, CIFAR-100-LT: tạo LT bằng quy luật 
𝑛
𝑦
=
𝑛
max
⁡
⋅
𝜌
𝑟
𝑦
𝐶
−
1
n
y
	​

=n
max
	​

⋅ρ
C−1
r
y
	​

	​

 (imbalance factor 
𝜌
∈
{
0.01
,
0.005
}
ρ∈{0.01,0.005} tương ứng IF=100/200).

ImageNet-LT, iNaturalist-LT: dùng split chính thức (không cần sinh LT).

1.2 Chia tập & “grouping”

Split: train / cal / val / test (ví dụ 80/10/5/5 cho CIFAR-LT; với ImageNet-LT dùng val chính thức và tách cal từ train).

Groups:

2-group (head/tail): tail = lớp có 
𝑛
𝑦
≤
𝜏
n
y
	​

≤τ (CIFAR-LT thường 
𝜏
∈
[
20
,
50
]
τ∈[20,50]).

3-group (head/medium/tail): phân vị 33%/66% theo 
𝑛
𝑦
n
y
	​

.

Lưu ánh xạ lớp → nhóm: 
[
𝑦
]
∈
{
1
,
…
,
𝐾
}
[y]∈{1,…,K}.

2) Tiền xử lý & sampling

Augment train: RandAugment (N=2, M=9–12), MixUp=0.2–0.4, CutMix=0.2–0.4, RandomResizedCrop, ColorJitter nhẹ; test/val dùng CenterCrop/Resize.

Class-Aware Sampling (CAS): chọn lớp uniform rồi lấy mẫu trong lớp; hoặc square-root sampling 
∝
𝑛
𝑦
∝
n
y
	​

	​

 nếu tail quá nhỏ.

Balanced batch: mỗi batch ≥ 50% ảnh từ non-head (medium+tail) để giảm “đè” head.

3) Huấn luyện base models (đa dạng hoá)

Chạy song song/tuần tự 3–6 mô hình:

Model #1 (cRT + Balanced Softmax)
Stage-1: CE instance-balanced 200–300 epochs; Stage-2: freeze backbone, retrain linear head với Balanced Softmax hoặc Logit-Adjust (+
log
⁡
𝜋
𝑦
logπ
y
	​

).

Model #2 (LDAM-DRW)
Margin 
𝑚
𝑦
=
𝐶
/
𝑛
𝑦
1
/
4
m
y
	​

=C/n
y
1/4
	​

; Deferred Re-Weighting nửa sau training.

Model #3 (CB-Focal)
Class-Balanced weight 
𝑤
𝑦
=
1
−
𝛽
1
−
𝛽
𝑛
𝑦
w
y
	​

=
1−β
n
y
	​

1−β
	​

 (
𝛽
β=0.999), Focal 
𝛾
=
1
∼
2
γ=1∼2.

(Tuỳ chọn) Kiến trúc khác (ConvNeXt/ViT), MixUp/CutMix/Res khác biệt, seeds khác để tăng diversity.

Hyper gợi ý (CIFAR-LT):
SGD-momentum 0.9, wd=5e-4, cosine LR 0.1→0, bs=256, epochs=200; EMA on. Với ImageNet-LT: bs=1024, epochs 180–300; RandAugment mạnh hơn; AMP fp16.

Kết xuất: Lưu logits & probs 
𝑝
(
𝑚
)
(
𝑥
)
p
(m)
(x) cho {cal, val, test} (định dạng npz/zarr), kèm id ảnh, nhãn, group.

4) Calibration theo nhóm

Trước khi ensemble/gating, calibrate từng model per-group (head vs tail):

Temperature scaling (per-group): tìm 
𝑇
𝑔
>
0
T
g
	​

>0 tối thiểu NLL trên cal-group 
𝑔
g.

(Tuỳ chọn) Vector scaling cho cụm lớp hay nhầm.

Áp 
𝑇
𝑔
T
g
	​

 khi suy ra 
𝑝
(
𝑚
)
(
𝑥
)
p
(m)
(x) trên mọi tập.

5) Gating 
𝑤
(
𝑥
;
𝜃
)
w(x;θ) & đặc trưng đầu vào

Đầu vào gating (trên cal/val): với mỗi ảnh 
𝑥
x,

concat 
[
𝑝
(
1
)
(
𝑥
)
,
…
,
𝑝
(
𝑀
)
(
𝑥
)
]
[p
(1)
(x),…,p
(M)
(x)] (MC dim 
𝑀
 ⁣
×
 ⁣
𝐶
M×C),

thêm meta-features: entropy từng model, max-prob từng model, variance/logit-disagreement, one-hot group 
[
𝑔
(
𝑥
)
]
[g(x)] hoặc extra bit “is-tail”.

Mạng gating: MLP 1–2 tầng (hidden 64–128), Softmax cuối để ra 
𝑤
(
𝑥
;
𝜃
)
∈
Δ
𝑀
−
1
w(x;θ)∈Δ
M−1
.
Ensemble posterior: 
𝑝
𝑄
𝜃
(
𝑥
)
=
∑
𝑚
𝑤
𝑚
(
𝑥
;
𝜃
)
 
𝑝
(
𝑚
)
(
𝑥
)
p
Q
θ
	​

	​

(x)=∑
m
	​

w
m
	​

(x;θ)p
(m)
(x).

6) Quy tắc plug-in (Theorem-1) & cố định-điểm 
𝛼
,
𝜇
α,μ

Cho chi phí reject 
𝑐
c. Với 
𝑝
𝑄
𝜃
p
Q
θ
	​

	​

, định nghĩa:

Classifier: 
ℎ
𝜃
(
𝑥
)
=
arg
⁡
max
⁡
𝑦
𝑝
𝑄
𝜃
,
𝑦
(
𝑥
)
𝛼
[
𝑦
]
h
θ
	​

(x)=argmax
y
	​

α
[y]
	​

p
Q
θ
	​

,y
	​

(x)
	​

.

Rejector: reject nếu

max
⁡
𝑦
𝑝
𝑄
𝜃
,
𝑦
(
𝑥
)
𝛼
[
𝑦
]
  
<
  
∑
𝑗
(
1
𝛼
[
𝑗
]
−
𝜇
[
𝑗
]
)
 
𝑝
𝑄
𝜃
,
𝑗
(
𝑥
)
  
−
  
𝑐
.
y
max
	​

α
[y]
	​

p
Q
θ
	​

,y
	​

(x)
	​

<
j
∑
	​

(
α
[j]
	​

1
	​

−μ
[j]
	​

)p
Q
θ
	​

,j
	​

(x)−c.

Cố định-điểm 
𝛼
α (trên cal):

Khởi tạo 
𝛼
𝑘
(
0
)
=
1
α
k
(0)
	​

=1 (hoặc theo tỉ lệ nhóm).

Cho trước 
𝜇
μ (xem dưới), lặp 
𝑡
=
0
→
𝑇
t=0→T (T≈10):

Dùng 
𝛼
(
𝑡
)
α
(t)
 suy ra quyết định accept/reject theo trên.

Cập nhật 
𝛼
𝑘
(
𝑡
+
1
)
←
𝐾
⋅
Pr
⁡
cal
(
𝑟
𝜃
(
𝑥
)
=
0
,
𝑦
∈
𝐺
𝑘
)
α
k
(t+1)
	​

←K⋅Pr
cal
	​

(r
θ
	​

(x)=0,y∈G
k
	​

).

Trả 
𝛼
^
α
^
 hội tụ.

Chọn 
𝜇
μ:

Với 
𝐾
=
2
K=2 (head/tail), đặt 
𝜇
=
[
𝜆
,
−
𝜆
]
μ=[λ,−λ]. Quét 
𝜆
∈
Λ
λ∈Λ (grid nhỏ, ví dụ 
[
−
2
,
2
]
[−2,2] step 0.1), mỗi 
𝜆
λ chạy cố định-điểm 
𝛼
α, chọn 
𝜆
^
λ
^
 tối thiểu balanced selective error (hoặc AURC) trên cal.

Với 
𝐾
>
2
K>2: ràng buộc 
∑
𝑘
𝜇
𝑘
=
0
∑
k
	​

μ
k
	​

=0 và dùng EG nhỏ 5–10 bước để giảm balanced error (giữ 
𝛼
α cập nhật nội vòng).

Lưu 
(
𝛼
^
,
𝜇
^
)
(
α
^
,
μ
^
	​

) để dùng ở val/test.

7) Huấn luyện gating bằng cận PAC-Bayes

Ta cần dataset S (độc lập cal) để tối ưu bound (tránh leak). Loss tuyến tính cân bằng (trên S):

ℓ
𝛼
^
(
𝜃
;
(
𝑥
,
𝑦
)
)
=
∑
𝑘
=
1
𝐾
1
𝛼
^
𝑘
 
1
[
𝑦
∈
𝐺
𝑘
]
⋅
1
[
𝑟
𝜃
(
𝑥
)
=
0
]
⋅
1
[
ℎ
𝜃
(
𝑥
)
≠
𝑦
]
+
𝑐
 
1
[
𝑟
𝜃
(
𝑥
)
=
1
]
.
ℓ
α
^
	​

(θ;(x,y))=
k=1
∑
K
	​

α
^
k
	​

1
	​

1[y∈G
k
	​

]⋅1[r
θ
	​

(x)=0]⋅1[h
θ
	​

(x)

=y]+c1[r
θ
	​

(x)=1].

Chuẩn hoá 
ℓ
~
∈
[
0
,
1
]
ℓ
~
∈[0,1] bằng 
𝐿
𝛼
^
=
max
⁡
{
𝑐
,
∑
𝑘
1
𝛼
^
𝑘
}
+
𝑐
L
α
^
	​

=max{c,∑
k
	​

α
^
k
	​

1
	​

}+c.
Bound cần tối thiểu (Deterministic hoặc Stochastic 
𝑄
Q):

𝐸
𝜃
∼
𝑄
 ⁣
[
1
∣
𝑆
∣
∑
(
𝑥
,
𝑦
)
∈
𝑆
ℓ
𝛼
^
(
𝜃
;
𝑥
,
𝑦
)
]
⏟
empirical
+
  
𝐿
𝛼
^
K
L
(
𝑄
∥
Π
)
+
ln
⁡
 ⁣
2
∣
𝑆
∣
𝛿
2
(
∣
𝑆
∣
−
1
)
  
+
  
𝜆
 
Ω
fair
(
𝑄
)
.
empirical
E
θ∼Q
	​

	​

∣S∣
1
	​

(x,y)∈S
∑
	​

ℓ
α
^
	​

(θ;x,y)
	​

	​

	​

+L
α
^
	​

2(∣S∣−1)
KL(Q∥Π)+ln
δ
2
∣S∣
	​

	​

	​

	​

+λΩ
fair
	​

(Q).

Hai mức triển khai:

Thực dụng (deterministic 
𝑄
=
𝛿
𝜃
Q=δ
θ
	​

): tối thiểu 
1
∣
𝑆
∣
∑
ℓ
𝛼
^
(
𝜃
)
+
𝛾
∥
𝜃
∥
2
2
∣S∣
1
	​

∑ℓ
α
^
	​

(θ)+γ∥θ∥
2
2
	​

 (coi 
K
L
KL ≈ L2). Nhanh, ổn định, dễ viết code.

Chuẩn PAC-Bayes (Gaussian 
𝑄
Q): 
𝑄
=
𝑁
(
𝜇
𝜃
,
𝜎
2
𝐼
)
Q=N(μ
θ
	​

,σ
2
I), 
Π
=
𝑁
(
0
,
𝜎
0
2
𝐼
)
Π=N(0,σ
0
2
	​

I). Dùng reparameterization trick để ước lượng 
𝐸
𝑄
[
ℓ
]
E
Q
	​

[ℓ]; thêm term 
K
L
(
𝑄
∥
Π
)
=
1
2
∑
𝑖
(
𝜎
2
+
𝜇
𝑖
2
𝜎
0
2
−
1
−
log
⁡
𝜎
2
𝜎
0
2
)
KL(Q∥Π)=
2
1
	​

∑
i
	​

(
σ
0
2
	​

σ
2
+μ
i
2
	​

	​

−1−log
σ
0
2
	​

σ
2
	​

).

Group-aware: tăng 
𝜎
0
σ
0
	​

 (prior variance) cho tail-features hoặc thêm ràng buộc budget KL theo group (thực thi bằng phạt 
∑
𝑔
𝜉
𝑔
 
K
L
𝑔
∑
g
	​

ξ
g
	​

KL
g
	​

).

Huấn luyện bằng Adam (lr 1e-3), batch 1024 entries (đọc cache 
𝑝
(
𝑚
)
(
𝑥
)
p
(m)
(x)), 10–30 epochs; early-stop theo balanced selective error trên val.

8) Mở rộng Worst-Group (ψ = max)

Vòng ngoài: trọng số nhóm 
𝛽
∈
Δ
𝐾
−
1
β∈Δ
K−1
 (khởi uniform). Lặp T=20–30:

Trong: thay 
1
𝛼
^
𝑘
α
^
k
	​

1
	​

 bằng 
𝛽
𝑘
⋅
1
𝛼
^
𝑘
β
k
	​

⋅
α
^
k
	​

1
	​

 trong 
ℓ
ℓ, huấn luyện gating như §7 (ngắn, 3–5 epochs fine-tune).

Tính lỗi nhóm 
𝑒
^
𝑘
e
^
k
	​

 trên val; cập nhật EG: 
𝛽
𝑘
∝
𝛽
𝑘
exp
⁡
(
𝜂
 
𝑒
^
𝑘
)
β
k
	​

∝β
k
	​

exp(η
e
^
k
	​

); chuẩn hoá 
∑
𝛽
𝑘
=
1
∑β
k
	​

=1.

Giữ mô hình/gating tốt nhất theo worst-group selective error.

9) Suy luận trên test

Với 
𝑝
(
𝑚
)
p
(m)
 (đã calibrate), chạy gating 
𝑤
(
𝑥
;
𝜃
)
w(x;θ) → 
𝑝
𝑄
𝜃
(
𝑥
)
p
Q
θ
	​

	​

(x).
Áp 
(
𝛼
^
,
𝜇
^
)
(
α
^
,
μ
^
	​

), 
𝑐
c, quyết định bằng Theorem-1 plug-in → nhãn hoặc reject.
Nếu làm worst-group: dùng 
𝛽
β cuối cùng (hoặc hỗn hợp iterate theo no-regret).

10) Metrics & cách tính

Giả sử trên tập 
𝑇
T có nhóm 
𝐺
𝑘
G
k
	​

, 
𝑘
=
1..
𝐾
k=1..K. Ký hiệu:

𝑟
(
𝑥
)
∈
{
0
,
1
}
r(x)∈{0,1} (1=reject), 
𝑦
^
(
𝑥
)
y
^
	​

(x) dự đoán khi 
𝑟
=
0
r=0.

Coverage: 
c
o
v
=
1
−
Pr
⁡
𝑇
(
𝑟
=
1
)
cov=1−Pr
T
	​

(r=1).
Balanced selective error:

B
S
E
=
1
𝐾
∑
𝑘
=
1
𝐾
Pr
⁡
𝑇
(
𝑦
^
≠
𝑦
∣
𝑟
=
0
,
 
𝑦
∈
𝐺
𝑘
)
.
BSE=
K
1
	​

k=1
∑
K
	​

T
Pr
	​

(
y
^
	​


=y∣r=0, y∈G
k
	​

).

Worst-group selective error: 
max
⁡
𝑘
Pr
⁡
𝑇
(
𝑦
^
≠
𝑦
∣
𝑟
=
0
,
𝑦
∈
𝐺
𝑘
)
max
k
	​

Pr
T
	​

(
y
^
	​


=y∣r=0,y∈G
k
	​

).
AURC: quét ngưỡng (hoặc “đòn bẩy” 
𝜇
μ/
𝑐
c) để vẽ Risk–Coverage (Risk = lỗi có điều kiện trên phần được chấp nhận), tính diện tích (trapezoid).
Accept-rate@group: 
Pr
⁡
𝑇
(
𝑟
=
0
∣
𝑦
∈
𝐺
𝑘
)
Pr
T
	​

(r=0∣y∈G
k
	​

).
ECE@group: ECE tính riêng cho mỗi nhóm (bining theo confidence 
max
⁡
𝑦
𝑝
𝑄
𝜃
,
𝑦
max
y
	​

p
Q
θ
	​

,y
	​

 trên phần được chấp nhận).

Khi báo cáo, cố định coverage (ví dụ 70%, 80%) và so sánh BSE/WGSE tại coverage đó; kèm full RC-curve & AURC.

11) Ablation bắt buộc (viết paper)

Deterministic vs Gaussian 
𝑄
Q; prior thường vs group-aware.

Có/không calibration theo nhóm.

Số mô hình M & đa dạng (loss/sampling/augment).

So sánh ghép: avg logits, log-opinion pool, PB-GSE, và PPRA (robust polytope) như biến thể.

Ảnh hưởng 
𝑐
c (0.05–0.5) và 
𝜇
μ/
𝛼
α (grid size, vòng cố định-điểm).

Worst-group EG on/off.

Head vs Tail: bảng Accept-rate@group, ECE@group, lỗi@coverage—chứng minh fair-ness cải thiện.

12) Tổ chức code & tái lập
pb-gse/
  configs/
    data.yaml        # đường dẫn, IF, seed, split tỉ lệ
    base_*.yaml      # cấu hình từng base model
    gating.yaml      # MLP, features, prior, KL weight, δ, ...
    plug_in.yaml     # c, grid λ, T_fixedpoint, K groups
  data/
    cifar_lt_indices.json
  models/
    backbones.py
    losses_lt.py     # LDAM-DRW, CB-Focal, BalancedSoftmax, ...
    calibration.py   # temp per-group, vector scaling
    gating.py        # MLP + features + (Gaussian Q) reparam
    plugin_rule.py   # Theorem-1 plug-in + fixed-point α, μ
  scripts/
    train_base.py
    dump_probs.py    # lưu p^{(m)}(x) cho cal/val/test
    calibrate.py
    train_gating_pacbayes.py
    infer_plugin.py
    eval_metrics.py  # BSE/WGSE/AURC/ECE@group
  outputs/
    probs/{model}/cal|val|test.npz
    gating/{run_id}/theta.pt
    plugin/{run_id}/alpha_mu.json
    metrics/{run_id}/*.json


Seed & log: cố định seed (torch/cuda/python), log Hydra/Weights&Biases, lưu hash git, thông số 
𝑐
c, 
𝜆
λ, 
𝛿
δ, KL-weight.

13) Pseudocode (các bước cốt lõi)

(A) Cố định-điểm 
𝛼
,
𝜇
α,μ (K=2)

def fixed_point_alpha_mu(PQ_cal, y_cal, group_cal, c, lambdas, T=10):
    # PQ_cal: [N, C] probs từ p_Q
    best = None
    for lam in lambdas:       # mu_head=lam, mu_tail=-lam
        mu = { 'head': lam, 'tail': -lam }
        alpha = {'head': 1.0, 'tail': 1.0}
        for _ in range(T):
            # compute decisions using current alpha, mu
            acc_mask = []   # r(x)=0
            err_mask = []   # r=0 & wrong
            grp_mask = {'head': [], 'tail': []}
            for i,(p, y, g) in enumerate(zip(PQ_cal, y_cal, group_cal)):
                lhs = (p / alpha[g_idx(y, g)]).max()
                rhs = ((1/alpha['head'] - mu['head']) * p[head_ids].sum()
                     + (1/alpha['tail'] - mu['tail']) * p[tail_ids].sum() - c)
                accept = lhs >= rhs
                acc_mask.append(accept)
                grp_mask[g].append((i, accept))
                if accept and (y != argmax_balanced(p, alpha)): err_mask.append(i)
            # update alpha
            for g in ['head','tail']:
                accept_in_g = sum(1 for i,acc in grp_mask[g] if acc)
                total_in_g   = sum(1 for i,_ in grp_mask[g])
                alpha[g] = 2.0 * (accept_in_g / max(1, total_in_g))
        # compute balanced selective error on cal
        bse = 0.5 * sum( err_rate_in_group(...) for g in ['head','tail'] )
        if (best is None) or (bse < best['bse']):
            best = {'alpha': alpha.copy(), 'mu': mu.copy(), 'bse': bse, 'lambda': lam}
    return best


(B) Loss tuyến tính cân bằng cho gating

def balanced_linear_loss(PQ, y, g, alpha, mu, c):
    # One sample (vectorized trong code thật)
    lhs = (PQ / alpha[g_idx_y_vector]).max()  # theo lớp
    rhs = ((1/alpha_grp - mu_grp) @ sum_probs_by_group(PQ)) - c
    accept = lhs >= rhs
    pred = argmax_balanced(PQ, alpha)
    err = (pred != y)
    return ((1/alpha[g]) * int(accept and err)) + c * int(not accept)


(C) Huấn luyện gating theo bound (deterministic)

for epoch in range(E):
    for batch in loader_S:
        feats = build_features(P_list_batch, groups_batch)
        w = gating(feats)                     # softmax over M
        PQ = sum_m( w[:,m].unsqueeze(-1) * P_m_batch[m] )
        loss_emp = mean( balanced_linear_loss(PQ, y, g, alpha_hat, mu_hat, c) )
        loss = loss_emp + gamma * l2(theta) + lambda_fair * fairness_penalty(...)
        loss.backward(); opt.step(); opt.zero_grad()
    # early stop theo BSE trên val

14) Báo cáo & hình vẽ (đưa vào paper)

Bảng chính: BSE@{70,80,90% coverage}, WGSE@{70,80%}, AURC (↓ tốt), Accept-rate@group (≈ cân), ECE@group (↓).

Hình: RC-curves theo nhóm; histogram confidence trước/sau calibration; map rejection trên simplex (CIFAR-10).

Ablation: (i) +calibration; (ii) +PAC-Bayes (vs L2); (iii) +worst-group EG; (iv) #models; (v) thay PB-GSE bằng PPRA/Avg.

Checklist “để nộp paper”

 Mã chạy end-to-end với config cố định, 3 seeds, std-error.

 Tiền xử lý rõ ràng, LT-indices public.

 Lưu artifact: 
𝑝
(
𝑚
)
p
(m)
, 
𝛼
^
,
𝜇
^
α
^
,
μ
^
	​

, 
𝜃
θ, logs.

 Báo cáo đủ metric & hình; so sánh baselines mạnh (paper gốc, DeepEns+Chow, Conformal).