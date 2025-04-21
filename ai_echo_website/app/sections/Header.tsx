"use client";

import Image from "next/image";
import Link from "next/link";

const Header = () => {
    return (
    <header className="fixed inset-x-0 top-0 z-20 bg-[#0B0B0F]/60 backdrop-blur-sm">
        <nav className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
        {/* --- brand ------------------------------------------------------- */}
        <Link href="/" className="flex items-center gap-2 font-semibold">
            {/* placeholder logo ------------------------------------------------*/}
            <div className="h-16 w-16 relative rounded-full overflow-hidden">
                        <Image
                            src="/logo.png"  // this assumes logo.png is inside /public
                            alt="AI Echo Logo"
                            fill
                            className="object-cover"
                        />
                    </div>
        </Link>

        {/* --- nav links ---------------------------------------------------- */}
        <ul className="hidden md:flex items-center gap-8 text-sm">
            {["About", "Contact"].map((item) => (
            <li key={item} className="relative">
                <a
                href="#"
                className="transition-colors hover:text-primary"
                >
                {item}
                </a>
            </li>
            ))}
        </ul>

        {/* --- call‑to‑action --------------------------------------------- */}
        </nav>
    </header>
    );
}

export default Header