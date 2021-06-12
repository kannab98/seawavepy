
pkgname=python-science-plots
pkgver='1.0.7'
pkgrel=1
arch=('x86_64', 'arm7h')
url='https://github.com/kannab98/seawavepy'
license=('MIT')
depends=('python-numpy', 'python-scipy', 'python-toml')
makedepends=('git')
source=("${pkgname}::git+${url}")
sha256sums=('SKIP')

build() {
  python ${srcdir}/${pkgname}/setup.py build
}

package() {
  python ${srcdir}/${pkgname}/setup.py install --root="$pkgdir" --optimize=1 --skip-build
  styledir="${pkgdir}/usr/lib/python3.9/site-packages/matplotlib/mpl-data/stylelib"

  mkdir -p ${styledir}
  files=$(find ${srcdir}/${pkgname}/styles -name "*.mplstyle")
  cp -r $files ${styledir}

}
# vim:set ts=2 sw=2 et:

