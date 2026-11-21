#!/bin/sh
# Called in the Python builder stage. The four wheels and their manifest can
# come from a private build context before the component commits are published.
set -eu
case "${MEDDEID_COMPONENT_SOURCE:-git}" in
  wheels)
    cd /component-wheels
    export LC_ALL=C
    sha256sum -- *.whl > /tmp/meddeid-component-hashes
    diff -u SHA256SUMS /tmp/meddeid-component-hashes
    python -m pip install --no-index --find-links . --constraint /tmp/container-constraints.txt \
      meddeid-core==0.2.4rc4 meddeid-language-en==0.2.4rc1 meddeid-language-nl==0.2.4rc1
    ;;
  git)
    python -m pip install --constraint /tmp/container-constraints.txt --retries 5 --timeout 60 \
      "meddeid-core @ git+https://github.com/stighellemans/meddeid-core.git@${MEDDEID_CORE_COMMIT}" \
      "meddeid-language-en @ git+https://github.com/stighellemans/meddeid-language-en.git@${MEDDEID_LANGUAGE_EN_COMMIT}" \
      "meddeid-language-nl @ git+https://github.com/stighellemans/meddeid-language-nl.git@${MEDDEID_LANGUAGE_NL_COMMIT}"
    ;;
  *) echo "MEDDEID_COMPONENT_SOURCE must be git or wheels" >&2; exit 2 ;;
esac
