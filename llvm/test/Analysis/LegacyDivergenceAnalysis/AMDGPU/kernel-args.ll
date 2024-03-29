; RUN: opt %s -mtriple amdgcn-- -passes='print<divergence>' 2>&1 -disable-output | FileCheck %s

; CHECK-LABEL: function 'test_amdgpu_ps':
; CHECK: DIVERGENT:  ptr addrspace(4) %arg0
; CHECK-NOT: DIVERGENT
; CHECK: DIVERGENT:  <2 x i32> %arg3
; CHECK: DIVERGENT:  <3 x i32> %arg4
; CHECK: DIVERGENT:  float %arg5
; CHECK: DIVERGENT:  i32 %arg6

define amdgpu_ps void @test_amdgpu_ps(ptr addrspace(4) byref([4 x <16 x i8>]) %arg0, float inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <3 x i32> %arg4, float %arg5, i32 %arg6) #0 {
  ret void
}

; CHECK-LABEL: function 'test_amdgpu_kernel':
; CHECK-NOT: %arg0
; CHECK-NOT: %arg1
; CHECK-NOT: %arg2
; CHECK-NOT: %arg3
; CHECK-NOT: %arg4
; CHECK-NOT: %arg5
; CHECK-NOT: %arg6
define amdgpu_kernel void @test_amdgpu_kernel(ptr addrspace(4) byref([4 x <16 x i8>]) %arg0, float inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <3 x i32> %arg4, float %arg5, i32 %arg6) #0 {
  ret void
}

; CHECK-LABEL: function 'test_c':
; CHECK: DIVERGENT:
; CHECK: DIVERGENT:
; CHECK: DIVERGENT:
; CHECK: DIVERGENT:
; CHECK: DIVERGENT:
; CHECK: DIVERGENT:
; CHECK: DIVERGENT:
define void @test_c(ptr addrspace(4) byval([4 x <16 x i8>]) %arg0, float inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <3 x i32> %arg4, float %arg5, i32 %arg6) #0 {
  ret void
}

attributes #0 = { nounwind }
