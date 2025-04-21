import React from 'react'

const Footer = () => {
    return (
        <footer className="border-t border-white/10 bg-[#0B0B0F] py-6 text-center text-sm text-gray-400">
          © {new Date().getFullYear()} AI Echo — All rights reserved.
        </footer>
    );
}

export default Footer