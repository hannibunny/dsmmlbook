У
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??

?
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
: *
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
: *
dtype0
?
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
: *
dtype0
?
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
:@*
dtype0
?
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
:@*
dtype0
?
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
:@*
dtype0
?
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:@*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
??*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:?*
dtype0
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	?
*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:
*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_10/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameRMSprop/conv2d_10/kernel/rms
?
0RMSprop/conv2d_10/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_10/kernel/rms*&
_output_shapes
: *
dtype0
?
RMSprop/conv2d_10/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv2d_10/bias/rms
?
.RMSprop/conv2d_10/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_10/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_11/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *-
shared_nameRMSprop/conv2d_11/kernel/rms
?
0RMSprop/conv2d_11/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_11/kernel/rms*&
_output_shapes
:  *
dtype0
?
RMSprop/conv2d_11/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv2d_11/bias/rms
?
.RMSprop/conv2d_11/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_11/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/conv2d_12/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*-
shared_nameRMSprop/conv2d_12/kernel/rms
?
0RMSprop/conv2d_12/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_12/kernel/rms*&
_output_shapes
: @*
dtype0
?
RMSprop/conv2d_12/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameRMSprop/conv2d_12/bias/rms
?
.RMSprop/conv2d_12/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_12/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/conv2d_13/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*-
shared_nameRMSprop/conv2d_13/kernel/rms
?
0RMSprop/conv2d_13/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_13/kernel/rms*&
_output_shapes
:@@*
dtype0
?
RMSprop/conv2d_13/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameRMSprop/conv2d_13/bias/rms
?
.RMSprop/conv2d_13/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_13/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/conv2d_14/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*-
shared_nameRMSprop/conv2d_14/kernel/rms
?
0RMSprop/conv2d_14/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_14/kernel/rms*&
_output_shapes
:@@*
dtype0
?
RMSprop/conv2d_14/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameRMSprop/conv2d_14/bias/rms
?
.RMSprop/conv2d_14/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_14/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/conv2d_15/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*-
shared_nameRMSprop/conv2d_15/kernel/rms
?
0RMSprop/conv2d_15/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_15/kernel/rms*&
_output_shapes
:@@*
dtype0
?
RMSprop/conv2d_15/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameRMSprop/conv2d_15/bias/rms
?
.RMSprop/conv2d_15/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_15/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/dense_10/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_nameRMSprop/dense_10/kernel/rms
?
/RMSprop/dense_10/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_10/kernel/rms* 
_output_shapes
:
??*
dtype0
?
RMSprop/dense_10/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameRMSprop/dense_10/bias/rms
?
-RMSprop/dense_10/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_10/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/dense_11/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*,
shared_nameRMSprop/dense_11/kernel/rms
?
/RMSprop/dense_11/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_11/kernel/rms*
_output_shapes
:	?
*
dtype0
?
RMSprop/dense_11/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_nameRMSprop/dense_11/bias/rms
?
-RMSprop/dense_11/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_11/bias/rms*
_output_shapes
:
*
dtype0

NoOpNoOp
?K
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?K
value?KB?K B?J
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
 trainable_variables
!regularization_losses
"	keras_api
h

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
R
/	variables
0trainable_variables
1regularization_losses
2	keras_api
h

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
h

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
R
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
R
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
h

Gkernel
Hbias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
h

Mkernel
Nbias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?
Siter
	Tdecay
Ulearning_rate
Vmomentum
Wrho
rms?
rms?
rms?
rms?
#rms?
$rms?
)rms?
*rms?
3rms?
4rms?
9rms?
:rms?
Grms?
Hrms?
Mrms?
Nrms?
v
0
1
2
3
#4
$5
)6
*7
38
49
910
:11
G12
H13
M14
N15
v
0
1
2
3
#4
$5
)6
*7
38
49
910
:11
G12
H13
M14
N15
 
?
	variables
Xmetrics
trainable_variables
Ylayer_metrics
Zlayer_regularization_losses
regularization_losses

[layers
\non_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
]metrics
trainable_variables
^layer_metrics
_layer_regularization_losses
regularization_losses

`layers
anon_trainable_variables
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
bmetrics
trainable_variables
clayer_metrics
dlayer_regularization_losses
regularization_losses

elayers
fnon_trainable_variables
 
 
 
?
	variables
gmetrics
 trainable_variables
hlayer_metrics
ilayer_regularization_losses
!regularization_losses

jlayers
knon_trainable_variables
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
%	variables
lmetrics
&trainable_variables
mlayer_metrics
nlayer_regularization_losses
'regularization_losses

olayers
pnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
?
+	variables
qmetrics
,trainable_variables
rlayer_metrics
slayer_regularization_losses
-regularization_losses

tlayers
unon_trainable_variables
 
 
 
?
/	variables
vmetrics
0trainable_variables
wlayer_metrics
xlayer_regularization_losses
1regularization_losses

ylayers
znon_trainable_variables
\Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
?
5	variables
{metrics
6trainable_variables
|layer_metrics
}layer_regularization_losses
7regularization_losses

~layers
non_trainable_variables
\Z
VARIABLE_VALUEconv2d_15/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_15/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1

90
:1
 
?
;	variables
?metrics
<trainable_variables
?layer_metrics
 ?layer_regularization_losses
=regularization_losses
?layers
?non_trainable_variables
 
 
 
?
?	variables
?metrics
@trainable_variables
?layer_metrics
 ?layer_regularization_losses
Aregularization_losses
?layers
?non_trainable_variables
 
 
 
?
C	variables
?metrics
Dtrainable_variables
?layer_metrics
 ?layer_regularization_losses
Eregularization_losses
?layers
?non_trainable_variables
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

G0
H1

G0
H1
 
?
I	variables
?metrics
Jtrainable_variables
?layer_metrics
 ?layer_regularization_losses
Kregularization_losses
?layers
?non_trainable_variables
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

M0
N1
 
?
O	variables
?metrics
Ptrainable_variables
?layer_metrics
 ?layer_regularization_losses
Qregularization_losses
?layers
?non_trainable_variables
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUERMSprop/conv2d_10/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_10/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_11/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_11/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_12/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_12/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_13/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_13/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_14/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_14/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_15/kernel/rmsTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/conv2d_15/bias/rmsRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense_10/kernel/rmsTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/dense_10/bias/rmsRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense_11/kernel/rmsTlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/dense_11/bias/rmsRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_10_inputPlaceholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_10_inputconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_106607
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0RMSprop/conv2d_10/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_10/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_11/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_11/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_12/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_12/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_13/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_13/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_14/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_14/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_15/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_15/bias/rms/Read/ReadVariableOp/RMSprop/dense_10/kernel/rms/Read/ReadVariableOp-RMSprop/dense_10/bias/rms/Read/ReadVariableOp/RMSprop/dense_11/kernel/rms/Read/ReadVariableOp-RMSprop/dense_11/bias/rms/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_107128
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1RMSprop/conv2d_10/kernel/rmsRMSprop/conv2d_10/bias/rmsRMSprop/conv2d_11/kernel/rmsRMSprop/conv2d_11/bias/rmsRMSprop/conv2d_12/kernel/rmsRMSprop/conv2d_12/bias/rmsRMSprop/conv2d_13/kernel/rmsRMSprop/conv2d_13/bias/rmsRMSprop/conv2d_14/kernel/rmsRMSprop/conv2d_14/bias/rmsRMSprop/conv2d_15/kernel/rmsRMSprop/conv2d_15/bias/rmsRMSprop/dense_10/kernel/rmsRMSprop/dense_10/bias/rmsRMSprop/dense_11/kernel/rmsRMSprop/dense_11/bias/rms*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_107261??
?

?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_106902

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_106822

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_6_layer_call_fn_106091

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1060852
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
-__inference_sequential_4_layer_call_fn_106774

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_1064402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????  ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

*__inference_conv2d_12_layer_call_fn_106871

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_1061732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?W
?

H__inference_sequential_4_layer_call_and_return_conditional_losses_106737

inputs,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_10/Relu?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_11/BiasAdd~
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_11/Relu?
max_pooling2d_5/MaxPoolMaxPoolconv2d_11/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPool?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2D max_pooling2d_5/MaxPool:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_12/Relu?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2Dconv2d_12/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_13/Relu?
max_pooling2d_6/MaxPoolMaxPoolconv2d_13/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2D max_pooling2d_6/MaxPool:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_14/BiasAdd~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_14/Relu?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Dconv2d_14/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_15/Relu?
max_pooling2d_7/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPools
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_5/Const?
flatten_5/ReshapeReshape max_pooling2d_7/MaxPool:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_5/Reshape?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMulflatten_5/Reshape:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_10/Relu?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_11/MatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_11/Softmax?
IdentityIdentitydense_11/Softmax:softmax:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????  ::::::::::::::::2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_106882

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
-__inference_sequential_4_layer_call_fn_106475
conv2d_10_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_1064402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????  ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_10_input
?	
?
D__inference_dense_10_layer_call_and_return_conditional_losses_106297

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?7
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_106341
conv2d_10_input
conv2d_10_106129
conv2d_10_106131
conv2d_11_106156
conv2d_11_106158
conv2d_12_106184
conv2d_12_106186
conv2d_13_106211
conv2d_13_106213
conv2d_14_106239
conv2d_14_106241
conv2d_15_106266
conv2d_15_106268
dense_10_106308
dense_10_106310
dense_11_106335
dense_11_106337
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputconv2d_10_106129conv2d_10_106131*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_1061182#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_106156conv2d_11_106158*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_1061452#
!conv2d_11/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1060732!
max_pooling2d_5/PartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_12_106184conv2d_12_106186*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_1061732#
!conv2d_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_106211conv2d_13_106213*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_1062002#
!conv2d_13/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1060852!
max_pooling2d_6/PartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_14_106239conv2d_14_106241*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_1062282#
!conv2d_14/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_106266conv2d_15_106268*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_1062552#
!conv2d_15/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1060972!
max_pooling2d_7/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_1062782
flatten_5/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_10_106308dense_10_106310*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_1062972"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_106335dense_11_106337*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_1063242"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????  ::::::::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_10_input
?
L
0__inference_max_pooling2d_5_layer_call_fn_106079

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1060732
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_106937

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
~
)__inference_dense_11_layer_call_fn_106982

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_1063242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_conv2d_11_layer_call_fn_106851

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_1061452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_7_layer_call_fn_106103

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1060972
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?n
?
!__inference__wrapped_model_106067
conv2d_10_input9
5sequential_4_conv2d_10_conv2d_readvariableop_resource:
6sequential_4_conv2d_10_biasadd_readvariableop_resource9
5sequential_4_conv2d_11_conv2d_readvariableop_resource:
6sequential_4_conv2d_11_biasadd_readvariableop_resource9
5sequential_4_conv2d_12_conv2d_readvariableop_resource:
6sequential_4_conv2d_12_biasadd_readvariableop_resource9
5sequential_4_conv2d_13_conv2d_readvariableop_resource:
6sequential_4_conv2d_13_biasadd_readvariableop_resource9
5sequential_4_conv2d_14_conv2d_readvariableop_resource:
6sequential_4_conv2d_14_biasadd_readvariableop_resource9
5sequential_4_conv2d_15_conv2d_readvariableop_resource:
6sequential_4_conv2d_15_biasadd_readvariableop_resource8
4sequential_4_dense_10_matmul_readvariableop_resource9
5sequential_4_dense_10_biasadd_readvariableop_resource8
4sequential_4_dense_11_matmul_readvariableop_resource9
5sequential_4_dense_11_biasadd_readvariableop_resource
identity??-sequential_4/conv2d_10/BiasAdd/ReadVariableOp?,sequential_4/conv2d_10/Conv2D/ReadVariableOp?-sequential_4/conv2d_11/BiasAdd/ReadVariableOp?,sequential_4/conv2d_11/Conv2D/ReadVariableOp?-sequential_4/conv2d_12/BiasAdd/ReadVariableOp?,sequential_4/conv2d_12/Conv2D/ReadVariableOp?-sequential_4/conv2d_13/BiasAdd/ReadVariableOp?,sequential_4/conv2d_13/Conv2D/ReadVariableOp?-sequential_4/conv2d_14/BiasAdd/ReadVariableOp?,sequential_4/conv2d_14/Conv2D/ReadVariableOp?-sequential_4/conv2d_15/BiasAdd/ReadVariableOp?,sequential_4/conv2d_15/Conv2D/ReadVariableOp?,sequential_4/dense_10/BiasAdd/ReadVariableOp?+sequential_4/dense_10/MatMul/ReadVariableOp?,sequential_4/dense_11/BiasAdd/ReadVariableOp?+sequential_4/dense_11/MatMul/ReadVariableOp?
,sequential_4/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_4/conv2d_10/Conv2D/ReadVariableOp?
sequential_4/conv2d_10/Conv2DConv2Dconv2d_10_input4sequential_4/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
sequential_4/conv2d_10/Conv2D?
-sequential_4/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_10/BiasAdd/ReadVariableOp?
sequential_4/conv2d_10/BiasAddBiasAdd&sequential_4/conv2d_10/Conv2D:output:05sequential_4/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2 
sequential_4/conv2d_10/BiasAdd?
sequential_4/conv2d_10/ReluRelu'sequential_4/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
sequential_4/conv2d_10/Relu?
,sequential_4/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_4/conv2d_11/Conv2D/ReadVariableOp?
sequential_4/conv2d_11/Conv2DConv2D)sequential_4/conv2d_10/Relu:activations:04sequential_4/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential_4/conv2d_11/Conv2D?
-sequential_4/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_4/conv2d_11/BiasAdd/ReadVariableOp?
sequential_4/conv2d_11/BiasAddBiasAdd&sequential_4/conv2d_11/Conv2D:output:05sequential_4/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2 
sequential_4/conv2d_11/BiasAdd?
sequential_4/conv2d_11/ReluRelu'sequential_4/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
sequential_4/conv2d_11/Relu?
$sequential_4/max_pooling2d_5/MaxPoolMaxPool)sequential_4/conv2d_11/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2&
$sequential_4/max_pooling2d_5/MaxPool?
,sequential_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02.
,sequential_4/conv2d_12/Conv2D/ReadVariableOp?
sequential_4/conv2d_12/Conv2DConv2D-sequential_4/max_pooling2d_5/MaxPool:output:04sequential_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential_4/conv2d_12/Conv2D?
-sequential_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_4/conv2d_12/BiasAdd/ReadVariableOp?
sequential_4/conv2d_12/BiasAddBiasAdd&sequential_4/conv2d_12/Conv2D:output:05sequential_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_4/conv2d_12/BiasAdd?
sequential_4/conv2d_12/ReluRelu'sequential_4/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_4/conv2d_12/Relu?
,sequential_4/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,sequential_4/conv2d_13/Conv2D/ReadVariableOp?
sequential_4/conv2d_13/Conv2DConv2D)sequential_4/conv2d_12/Relu:activations:04sequential_4/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential_4/conv2d_13/Conv2D?
-sequential_4/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_4/conv2d_13/BiasAdd/ReadVariableOp?
sequential_4/conv2d_13/BiasAddBiasAdd&sequential_4/conv2d_13/Conv2D:output:05sequential_4/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_4/conv2d_13/BiasAdd?
sequential_4/conv2d_13/ReluRelu'sequential_4/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_4/conv2d_13/Relu?
$sequential_4/max_pooling2d_6/MaxPoolMaxPool)sequential_4/conv2d_13/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2&
$sequential_4/max_pooling2d_6/MaxPool?
,sequential_4/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,sequential_4/conv2d_14/Conv2D/ReadVariableOp?
sequential_4/conv2d_14/Conv2DConv2D-sequential_4/max_pooling2d_6/MaxPool:output:04sequential_4/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
sequential_4/conv2d_14/Conv2D?
-sequential_4/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_4/conv2d_14/BiasAdd/ReadVariableOp?
sequential_4/conv2d_14/BiasAddBiasAdd&sequential_4/conv2d_14/Conv2D:output:05sequential_4/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_4/conv2d_14/BiasAdd?
sequential_4/conv2d_14/ReluRelu'sequential_4/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_4/conv2d_14/Relu?
,sequential_4/conv2d_15/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,sequential_4/conv2d_15/Conv2D/ReadVariableOp?
sequential_4/conv2d_15/Conv2DConv2D)sequential_4/conv2d_14/Relu:activations:04sequential_4/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential_4/conv2d_15/Conv2D?
-sequential_4/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_4/conv2d_15/BiasAdd/ReadVariableOp?
sequential_4/conv2d_15/BiasAddBiasAdd&sequential_4/conv2d_15/Conv2D:output:05sequential_4/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
sequential_4/conv2d_15/BiasAdd?
sequential_4/conv2d_15/ReluRelu'sequential_4/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
sequential_4/conv2d_15/Relu?
$sequential_4/max_pooling2d_7/MaxPoolMaxPool)sequential_4/conv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2&
$sequential_4/max_pooling2d_7/MaxPool?
sequential_4/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
sequential_4/flatten_5/Const?
sequential_4/flatten_5/ReshapeReshape-sequential_4/max_pooling2d_7/MaxPool:output:0%sequential_4/flatten_5/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential_4/flatten_5/Reshape?
+sequential_4/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_4/dense_10/MatMul/ReadVariableOp?
sequential_4/dense_10/MatMulMatMul'sequential_4/flatten_5/Reshape:output:03sequential_4/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_4/dense_10/MatMul?
,sequential_4/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_4/dense_10/BiasAdd/ReadVariableOp?
sequential_4/dense_10/BiasAddBiasAdd&sequential_4/dense_10/MatMul:product:04sequential_4/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_4/dense_10/BiasAdd?
sequential_4/dense_10/ReluRelu&sequential_4/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_4/dense_10/Relu?
+sequential_4/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_11_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02-
+sequential_4/dense_11/MatMul/ReadVariableOp?
sequential_4/dense_11/MatMulMatMul(sequential_4/dense_10/Relu:activations:03sequential_4/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_4/dense_11/MatMul?
,sequential_4/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,sequential_4/dense_11/BiasAdd/ReadVariableOp?
sequential_4/dense_11/BiasAddBiasAdd&sequential_4/dense_11/MatMul:product:04sequential_4/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_4/dense_11/BiasAdd?
sequential_4/dense_11/SoftmaxSoftmax&sequential_4/dense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
sequential_4/dense_11/Softmax?
IdentityIdentity'sequential_4/dense_11/Softmax:softmax:0.^sequential_4/conv2d_10/BiasAdd/ReadVariableOp-^sequential_4/conv2d_10/Conv2D/ReadVariableOp.^sequential_4/conv2d_11/BiasAdd/ReadVariableOp-^sequential_4/conv2d_11/Conv2D/ReadVariableOp.^sequential_4/conv2d_12/BiasAdd/ReadVariableOp-^sequential_4/conv2d_12/Conv2D/ReadVariableOp.^sequential_4/conv2d_13/BiasAdd/ReadVariableOp-^sequential_4/conv2d_13/Conv2D/ReadVariableOp.^sequential_4/conv2d_14/BiasAdd/ReadVariableOp-^sequential_4/conv2d_14/Conv2D/ReadVariableOp.^sequential_4/conv2d_15/BiasAdd/ReadVariableOp-^sequential_4/conv2d_15/Conv2D/ReadVariableOp-^sequential_4/dense_10/BiasAdd/ReadVariableOp,^sequential_4/dense_10/MatMul/ReadVariableOp-^sequential_4/dense_11/BiasAdd/ReadVariableOp,^sequential_4/dense_11/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????  ::::::::::::::::2^
-sequential_4/conv2d_10/BiasAdd/ReadVariableOp-sequential_4/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_10/Conv2D/ReadVariableOp,sequential_4/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_11/BiasAdd/ReadVariableOp-sequential_4/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_11/Conv2D/ReadVariableOp,sequential_4/conv2d_11/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_12/BiasAdd/ReadVariableOp-sequential_4/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_12/Conv2D/ReadVariableOp,sequential_4/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_13/BiasAdd/ReadVariableOp-sequential_4/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_13/Conv2D/ReadVariableOp,sequential_4/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_14/BiasAdd/ReadVariableOp-sequential_4/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_14/Conv2D/ReadVariableOp,sequential_4/conv2d_14/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_15/BiasAdd/ReadVariableOp-sequential_4/conv2d_15/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_15/Conv2D/ReadVariableOp,sequential_4/conv2d_15/Conv2D/ReadVariableOp2\
,sequential_4/dense_10/BiasAdd/ReadVariableOp,sequential_4/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_10/MatMul/ReadVariableOp+sequential_4/dense_10/MatMul/ReadVariableOp2\
,sequential_4/dense_11/BiasAdd/ReadVariableOp,sequential_4/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_11/MatMul/ReadVariableOp+sequential_4/dense_11/MatMul/ReadVariableOp:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_10_input
?
F
*__inference_flatten_5_layer_call_fn_106942

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_1062782
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?W
?
__inference__traced_save_107128
file_prefix/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_rmsprop_conv2d_10_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_10_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_11_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_11_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_12_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_12_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_13_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_13_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_14_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_14_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_15_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_15_bias_rms_read_readvariableop:
6savev2_rmsprop_dense_10_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_10_bias_rms_read_readvariableop:
6savev2_rmsprop_dense_11_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_11_bias_rms_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_rmsprop_conv2d_10_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_10_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_11_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_11_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_12_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_12_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_13_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_13_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_14_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_14_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_15_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_15_bias_rms_read_readvariableop6savev2_rmsprop_dense_10_kernel_rms_read_readvariableop4savev2_rmsprop_dense_10_bias_rms_read_readvariableop6savev2_rmsprop_dense_11_kernel_rms_read_readvariableop4savev2_rmsprop_dense_11_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : :  : : @:@:@@:@:@@:@:@@:@:
??:?:	?
:
: : : : : : : : : : : :  : : @:@:@@:@:@@:@:@@:@:
??:?:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,	(
&
_output_shapes
:@@: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:, (
&
_output_shapes
:@@: !

_output_shapes
:@:,"(
&
_output_shapes
:@@: #

_output_shapes
:@:,$(
&
_output_shapes
:@@: %

_output_shapes
:@:&&"
 
_output_shapes
:
??:!'

_output_shapes	
:?:%(!

_output_shapes
:	?
: )

_output_shapes
:
:*

_output_shapes
: 
?7
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_106525

inputs
conv2d_10_106480
conv2d_10_106482
conv2d_11_106485
conv2d_11_106487
conv2d_12_106491
conv2d_12_106493
conv2d_13_106496
conv2d_13_106498
conv2d_14_106502
conv2d_14_106504
conv2d_15_106507
conv2d_15_106509
dense_10_106514
dense_10_106516
dense_11_106519
dense_11_106521
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_106480conv2d_10_106482*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_1061182#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_106485conv2d_11_106487*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_1061452#
!conv2d_11/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1060732!
max_pooling2d_5/PartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_12_106491conv2d_12_106493*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_1061732#
!conv2d_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_106496conv2d_13_106498*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_1062002#
!conv2d_13/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1060852!
max_pooling2d_6/PartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_14_106502conv2d_14_106504*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_1062282#
!conv2d_14/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_106507conv2d_15_106509*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_1062552#
!conv2d_15/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1060972!
max_pooling2d_7/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_1062782
flatten_5/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_10_106514dense_10_106516*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_1062972"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_106519dense_11_106521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_1063242"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????  ::::::::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

*__inference_conv2d_14_layer_call_fn_106911

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_1062282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

*__inference_conv2d_15_layer_call_fn_106931

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_1062552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_106255

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_106862

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
~
)__inference_dense_10_layer_call_fn_106962

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_1062972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_106118

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_106842

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?

?
-__inference_sequential_4_layer_call_fn_106811

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_1065252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????  ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_106922

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_106145

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?7
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_106389
conv2d_10_input
conv2d_10_106344
conv2d_10_106346
conv2d_11_106349
conv2d_11_106351
conv2d_12_106355
conv2d_12_106357
conv2d_13_106360
conv2d_13_106362
conv2d_14_106366
conv2d_14_106368
conv2d_15_106371
conv2d_15_106373
dense_10_106378
dense_10_106380
dense_11_106383
dense_11_106385
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputconv2d_10_106344conv2d_10_106346*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_1061182#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_106349conv2d_11_106351*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_1061452#
!conv2d_11/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1060732!
max_pooling2d_5/PartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_12_106355conv2d_12_106357*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_1061732#
!conv2d_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_106360conv2d_13_106362*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_1062002#
!conv2d_13/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1060852!
max_pooling2d_6/PartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_14_106366conv2d_14_106368*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_1062282#
!conv2d_14/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_106371conv2d_15_106373*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_1062552#
!conv2d_15/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1060972!
max_pooling2d_7/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_1062782
flatten_5/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_10_106378dense_10_106380*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_1062972"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_106383dense_11_106385*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_1063242"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????  ::::::::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_10_input
?

?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_106200

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_106228

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_106173

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense_10_layer_call_and_return_conditional_losses_106953

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_11_layer_call_and_return_conditional_losses_106324

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_106085

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_5_layer_call_and_return_conditional_losses_106278

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_107261
file_prefix%
!assignvariableop_conv2d_10_kernel%
!assignvariableop_1_conv2d_10_bias'
#assignvariableop_2_conv2d_11_kernel%
!assignvariableop_3_conv2d_11_bias'
#assignvariableop_4_conv2d_12_kernel%
!assignvariableop_5_conv2d_12_bias'
#assignvariableop_6_conv2d_13_kernel%
!assignvariableop_7_conv2d_13_bias'
#assignvariableop_8_conv2d_14_kernel%
!assignvariableop_9_conv2d_14_bias(
$assignvariableop_10_conv2d_15_kernel&
"assignvariableop_11_conv2d_15_bias'
#assignvariableop_12_dense_10_kernel%
!assignvariableop_13_dense_10_bias'
#assignvariableop_14_dense_11_kernel%
!assignvariableop_15_dense_11_bias$
 assignvariableop_16_rmsprop_iter%
!assignvariableop_17_rmsprop_decay-
)assignvariableop_18_rmsprop_learning_rate(
$assignvariableop_19_rmsprop_momentum#
assignvariableop_20_rmsprop_rho
assignvariableop_21_total
assignvariableop_22_count
assignvariableop_23_total_1
assignvariableop_24_count_14
0assignvariableop_25_rmsprop_conv2d_10_kernel_rms2
.assignvariableop_26_rmsprop_conv2d_10_bias_rms4
0assignvariableop_27_rmsprop_conv2d_11_kernel_rms2
.assignvariableop_28_rmsprop_conv2d_11_bias_rms4
0assignvariableop_29_rmsprop_conv2d_12_kernel_rms2
.assignvariableop_30_rmsprop_conv2d_12_bias_rms4
0assignvariableop_31_rmsprop_conv2d_13_kernel_rms2
.assignvariableop_32_rmsprop_conv2d_13_bias_rms4
0assignvariableop_33_rmsprop_conv2d_14_kernel_rms2
.assignvariableop_34_rmsprop_conv2d_14_bias_rms4
0assignvariableop_35_rmsprop_conv2d_15_kernel_rms2
.assignvariableop_36_rmsprop_conv2d_15_bias_rms3
/assignvariableop_37_rmsprop_dense_10_kernel_rms1
-assignvariableop_38_rmsprop_dense_10_bias_rms3
/assignvariableop_39_rmsprop_dense_11_kernel_rms1
-assignvariableop_40_rmsprop_dense_11_bias_rms
identity_42??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_11_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_11_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_12_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_12_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_13_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_13_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_14_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_14_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_15_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_15_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_10_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_10_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_11_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_11_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp assignvariableop_16_rmsprop_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_rmsprop_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_rmsprop_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_rmsprop_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_rmsprop_rhoIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_rmsprop_conv2d_10_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_rmsprop_conv2d_10_bias_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp0assignvariableop_27_rmsprop_conv2d_11_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp.assignvariableop_28_rmsprop_conv2d_11_bias_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp0assignvariableop_29_rmsprop_conv2d_12_kernel_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp.assignvariableop_30_rmsprop_conv2d_12_bias_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp0assignvariableop_31_rmsprop_conv2d_13_kernel_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp.assignvariableop_32_rmsprop_conv2d_13_bias_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp0assignvariableop_33_rmsprop_conv2d_14_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp.assignvariableop_34_rmsprop_conv2d_14_bias_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp0assignvariableop_35_rmsprop_conv2d_15_kernel_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp.assignvariableop_36_rmsprop_conv2d_15_bias_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp/assignvariableop_37_rmsprop_dense_10_kernel_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp-assignvariableop_38_rmsprop_dense_10_bias_rmsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp/assignvariableop_39_rmsprop_dense_11_kernel_rmsIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp-assignvariableop_40_rmsprop_dense_11_bias_rmsIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41?
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
$__inference_signature_wrapper_106607
conv2d_10_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1060672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????  ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_10_input
?W
?

H__inference_sequential_4_layer_call_and_return_conditional_losses_106672

inputs,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp? conv2d_12/BiasAdd/ReadVariableOp?conv2d_12/Conv2D/ReadVariableOp? conv2d_13/BiasAdd/ReadVariableOp?conv2d_13/Conv2D/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_10/Relu?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_11/BiasAdd~
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_11/Relu?
max_pooling2d_5/MaxPoolMaxPoolconv2d_11/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPool?
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_12/Conv2D/ReadVariableOp?
conv2d_12/Conv2DConv2D max_pooling2d_5/MaxPool:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_12/Conv2D?
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp?
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_12/BiasAdd~
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_12/Relu?
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_13/Conv2D/ReadVariableOp?
conv2d_13/Conv2DConv2Dconv2d_12/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_13/Conv2D?
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp?
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_13/Relu?
max_pooling2d_6/MaxPoolMaxPoolconv2d_13/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2D max_pooling2d_6/MaxPool:output:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_14/BiasAdd~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_14/Relu?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Dconv2d_14/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_15/Relu?
max_pooling2d_7/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPools
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_5/Const?
flatten_5/ReshapeReshape max_pooling2d_7/MaxPool:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_5/Reshape?
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_10/MatMul/ReadVariableOp?
dense_10/MatMulMatMulflatten_5/Reshape:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_10/MatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_10/BiasAddt
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_10/Relu?
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_11/MatMul/ReadVariableOp?
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_11/MatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_11/BiasAdd|
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_11/Softmax?
IdentityIdentitydense_11/Softmax:softmax:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????  ::::::::::::::::2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_106073

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_11_layer_call_and_return_conditional_losses_106973

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_conv2d_10_layer_call_fn_106831

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_1061182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
-__inference_sequential_4_layer_call_fn_106560
conv2d_10_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_1065252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????  ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????  
)
_user_specified_nameconv2d_10_input
?

*__inference_conv2d_13_layer_call_fn_106891

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_1062002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?7
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_106440

inputs
conv2d_10_106395
conv2d_10_106397
conv2d_11_106400
conv2d_11_106402
conv2d_12_106406
conv2d_12_106408
conv2d_13_106411
conv2d_13_106413
conv2d_14_106417
conv2d_14_106419
conv2d_15_106422
conv2d_15_106424
dense_10_106429
dense_10_106431
dense_11_106434
dense_11_106436
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall?!conv2d_12/StatefulPartitionedCall?!conv2d_13/StatefulPartitionedCall?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_106395conv2d_10_106397*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_1061182#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_106400conv2d_11_106402*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_1061452#
!conv2d_11/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_1060732!
max_pooling2d_5/PartitionedCall?
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_12_106406conv2d_12_106408*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_1061732#
!conv2d_12/StatefulPartitionedCall?
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_106411conv2d_13_106413*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_1062002#
!conv2d_13/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_1060852!
max_pooling2d_6/PartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_14_106417conv2d_14_106419*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_1062282#
!conv2d_14/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_106422conv2d_15_106424*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_1062552#
!conv2d_15/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_1060972!
max_pooling2d_7/PartitionedCall?
flatten_5/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_5_layer_call_and_return_conditional_losses_1062782
flatten_5/PartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_10_106429dense_10_106431*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_10_layer_call_and_return_conditional_losses_1062972"
 dense_10/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_106434dense_11_106436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_11_layer_call_and_return_conditional_losses_1063242"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????  ::::::::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_106097

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
conv2d_10_input@
!serving_default_conv2d_10_input:0?????????  <
dense_110
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?m
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?h
_tf_keras_sequential?h{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
?	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 32]}}
?	

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 64]}}
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 64]}}
?	

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 64]}}
?
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Gkernel
Hbias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Mkernel
Nbias
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
Siter
	Tdecay
Ulearning_rate
Vmomentum
Wrho
rms?
rms?
rms?
rms?
#rms?
$rms?
)rms?
*rms?
3rms?
4rms?
9rms?
:rms?
Grms?
Hrms?
Mrms?
Nrms?"
	optimizer
?
0
1
2
3
#4
$5
)6
*7
38
49
910
:11
G12
H13
M14
N15"
trackable_list_wrapper
?
0
1
2
3
#4
$5
)6
*7
38
49
910
:11
G12
H13
M14
N15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Xmetrics
trainable_variables
Ylayer_metrics
Zlayer_regularization_losses
regularization_losses

[layers
\non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_10/kernel
: 2conv2d_10/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
]metrics
trainable_variables
^layer_metrics
_layer_regularization_losses
regularization_losses

`layers
anon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_11/kernel
: 2conv2d_11/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
bmetrics
trainable_variables
clayer_metrics
dlayer_regularization_losses
regularization_losses

elayers
fnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
gmetrics
 trainable_variables
hlayer_metrics
ilayer_regularization_losses
!regularization_losses

jlayers
knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_12/kernel
:@2conv2d_12/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
%	variables
lmetrics
&trainable_variables
mlayer_metrics
nlayer_regularization_losses
'regularization_losses

olayers
pnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_13/kernel
:@2conv2d_13/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
+	variables
qmetrics
,trainable_variables
rlayer_metrics
slayer_regularization_losses
-regularization_losses

tlayers
unon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
/	variables
vmetrics
0trainable_variables
wlayer_metrics
xlayer_regularization_losses
1regularization_losses

ylayers
znon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_14/kernel
:@2conv2d_14/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
5	variables
{metrics
6trainable_variables
|layer_metrics
}layer_regularization_losses
7regularization_losses

~layers
non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_15/kernel
:@2conv2d_15/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;	variables
?metrics
<trainable_variables
?layer_metrics
 ?layer_regularization_losses
=regularization_losses
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?metrics
@trainable_variables
?layer_metrics
 ?layer_regularization_losses
Aregularization_losses
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
C	variables
?metrics
Dtrainable_variables
?layer_metrics
 ?layer_regularization_losses
Eregularization_losses
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_10/kernel
:?2dense_10/bias
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
I	variables
?metrics
Jtrainable_variables
?layer_metrics
 ?layer_regularization_losses
Kregularization_losses
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?
2dense_11/kernel
:
2dense_11/bias
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
O	variables
?metrics
Ptrainable_variables
?layer_metrics
 ?layer_regularization_losses
Qregularization_losses
?layers
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
4:2 2RMSprop/conv2d_10/kernel/rms
&:$ 2RMSprop/conv2d_10/bias/rms
4:2  2RMSprop/conv2d_11/kernel/rms
&:$ 2RMSprop/conv2d_11/bias/rms
4:2 @2RMSprop/conv2d_12/kernel/rms
&:$@2RMSprop/conv2d_12/bias/rms
4:2@@2RMSprop/conv2d_13/kernel/rms
&:$@2RMSprop/conv2d_13/bias/rms
4:2@@2RMSprop/conv2d_14/kernel/rms
&:$@2RMSprop/conv2d_14/bias/rms
4:2@@2RMSprop/conv2d_15/kernel/rms
&:$@2RMSprop/conv2d_15/bias/rms
-:+
??2RMSprop/dense_10/kernel/rms
&:$?2RMSprop/dense_10/bias/rms
,:*	?
2RMSprop/dense_11/kernel/rms
%:#
2RMSprop/dense_11/bias/rms
?2?
!__inference__wrapped_model_106067?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *6?3
1?.
conv2d_10_input?????????  
?2?
-__inference_sequential_4_layer_call_fn_106774
-__inference_sequential_4_layer_call_fn_106811
-__inference_sequential_4_layer_call_fn_106475
-__inference_sequential_4_layer_call_fn_106560?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_4_layer_call_and_return_conditional_losses_106341
H__inference_sequential_4_layer_call_and_return_conditional_losses_106672
H__inference_sequential_4_layer_call_and_return_conditional_losses_106389
H__inference_sequential_4_layer_call_and_return_conditional_losses_106737?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_10_layer_call_fn_106831?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_106822?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_11_layer_call_fn_106851?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_106842?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_5_layer_call_fn_106079?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_106073?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_conv2d_12_layer_call_fn_106871?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_106862?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_13_layer_call_fn_106891?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_106882?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_6_layer_call_fn_106091?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_106085?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_conv2d_14_layer_call_fn_106911?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_106902?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_15_layer_call_fn_106931?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_106922?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_7_layer_call_fn_106103?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_106097?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_flatten_5_layer_call_fn_106942?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_5_layer_call_and_return_conditional_losses_106937?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_10_layer_call_fn_106962?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_10_layer_call_and_return_conditional_losses_106953?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_11_layer_call_fn_106982?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_11_layer_call_and_return_conditional_losses_106973?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_106607conv2d_10_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_106067?#$)*349:GHMN@?=
6?3
1?.
conv2d_10_input?????????  
? "3?0
.
dense_11"?
dense_11?????????
?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_106822l7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????   
? ?
*__inference_conv2d_10_layer_call_fn_106831_7?4
-?*
(?%
inputs?????????  
? " ??????????   ?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_106842l7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_11_layer_call_fn_106851_7?4
-?*
(?%
inputs?????????   
? " ?????????? ?
E__inference_conv2d_12_layer_call_and_return_conditional_losses_106862l#$7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_12_layer_call_fn_106871_#$7?4
-?*
(?%
inputs????????? 
? " ??????????@?
E__inference_conv2d_13_layer_call_and_return_conditional_losses_106882l)*7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_13_layer_call_fn_106891_)*7?4
-?*
(?%
inputs?????????@
? " ??????????@?
E__inference_conv2d_14_layer_call_and_return_conditional_losses_106902l347?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_14_layer_call_fn_106911_347?4
-?*
(?%
inputs?????????@
? " ??????????@?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_106922l9:7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_15_layer_call_fn_106931_9:7?4
-?*
(?%
inputs?????????@
? " ??????????@?
D__inference_dense_10_layer_call_and_return_conditional_losses_106953^GH0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_10_layer_call_fn_106962QGH0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_11_layer_call_and_return_conditional_losses_106973]MN0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? }
)__inference_dense_11_layer_call_fn_106982PMN0?-
&?#
!?
inputs??????????
? "??????????
?
E__inference_flatten_5_layer_call_and_return_conditional_losses_106937a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
*__inference_flatten_5_layer_call_fn_106942T7?4
-?*
(?%
inputs?????????@
? "????????????
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_106073?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_5_layer_call_fn_106079?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_106085?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_6_layer_call_fn_106091?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_106097?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_7_layer_call_fn_106103?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_sequential_4_layer_call_and_return_conditional_losses_106341?#$)*349:GHMNH?E
>?;
1?.
conv2d_10_input?????????  
p

 
? "%?"
?
0?????????

? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_106389?#$)*349:GHMNH?E
>?;
1?.
conv2d_10_input?????????  
p 

 
? "%?"
?
0?????????

? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_106672z#$)*349:GHMN??<
5?2
(?%
inputs?????????  
p

 
? "%?"
?
0?????????

? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_106737z#$)*349:GHMN??<
5?2
(?%
inputs?????????  
p 

 
? "%?"
?
0?????????

? ?
-__inference_sequential_4_layer_call_fn_106475v#$)*349:GHMNH?E
>?;
1?.
conv2d_10_input?????????  
p

 
? "??????????
?
-__inference_sequential_4_layer_call_fn_106560v#$)*349:GHMNH?E
>?;
1?.
conv2d_10_input?????????  
p 

 
? "??????????
?
-__inference_sequential_4_layer_call_fn_106774m#$)*349:GHMN??<
5?2
(?%
inputs?????????  
p

 
? "??????????
?
-__inference_sequential_4_layer_call_fn_106811m#$)*349:GHMN??<
5?2
(?%
inputs?????????  
p 

 
? "??????????
?
$__inference_signature_wrapper_106607?#$)*349:GHMNS?P
? 
I?F
D
conv2d_10_input1?.
conv2d_10_input?????????  "3?0
.
dense_11"?
dense_11?????????
