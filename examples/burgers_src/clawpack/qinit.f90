subroutine qinit(meqn,mbc,mx,xlower,dx,q,maux,aux)

    ! Set initial conditions for the q array.
    ! This default version prints an error message since it should
    ! not be used directly.  Copy this to an application directory and
    ! loop over all grid cells to set values of q(1:meqn, 1:mx).

    implicit none
    
    integer, intent(in) :: meqn,mbc,mx,maux
    real(kind=8), intent(in) :: xlower,dx
    real(kind=8), intent(in) :: aux(maux,1-mbc:mx+mbc)
    real(kind=8), intent(inout) :: q(meqn,1-mbc:mx+mbc)

    integer :: i
    real(kind=8) :: r ,pi, xcell, mu1, mu2, xcenter,w

    common /cparam/ mu1,mu2

    pi = 4.0d0*datan(1.0d0)
    r = mu2
    xcenter = 0.0d0
    w = 2.0d0
    
    do i=1,mx
      xcell = xlower + (i-0.5d0)*dx
      if (dabs(xcell - xcenter) .le. w) then 
        q(1,i) = 0.5 + 0.5d0*dsin(pi*(0.5d0*(xcell - xcenter)/w - 1.0d0)) 
      else if ((xcell - xcenter) .gt. w) then 
        q(1,i) = 0.0d0 
      else if ((xcell - xcenter) .lt. w) then 
        q(1,i) = 1.0d0 
      endif
    enddo

end subroutine qinit

