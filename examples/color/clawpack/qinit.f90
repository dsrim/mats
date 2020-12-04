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
    real(kind=8) :: r ,pi, xcell, center, mu1,mu2,mu3,mu4

    common /cparam/ mu1,mu2,mu3,mu4

    pi = 4.0d0*datan(1.0d0)
    r = 0.2d0 
    center = 0.25d0+ 1.0d-8
    
    do i=1,mx
      xcell = xlower + (i-0.5d0)*dx
      if ((xcell .ge. center-r) .and. (xcell .le. center+r)) then
        q(1,i) = 0.5d0 + 0.5d0*dcos((xcell-center)/r*pi)
      elseif (xcell .le. center-r) then
        q(1,i) = 0.0d0
      else
        q(1,i) = 0.0d0
      endif
    enddo

end subroutine qinit


