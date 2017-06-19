import hashlib

if __name__ == "__main__":
    # MD5
    md5 = hashlib.md5()
    md5.update('string')
    print u'Messy code: ', md5.digest()
    print u'MD5: ', md5.hexdigest()
    print u'Digest Size: ', md5.digest_size
    print u'Block Size: ', md5.block_size

    # sha1
    sha1 = hashlib.sha1()
    sha1.update('string')
    print u'Messy code: ', sha1.digest()
    print u'SHA1: ', sha1.hexdigest()
    print u'Digest Size: ', sha1.digest_size
    print u'Block Size: ', sha1.block_size

    md5 = hashlib.new('md5', 'string')
    print md5.hexdigest()
    sha1 = hashlib.new('sha1', 'string')
    print sha1.hexdigest()

    # 'md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512'
    print hashlib.algorithms
