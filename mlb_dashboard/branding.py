from __future__ import annotations

import base64
from pathlib import Path
from functools import lru_cache

import streamlit as st
import streamlit.components.v1 as components


def page_icon_path() -> str:
    return str(Path(__file__).resolve().parent / "assets" / "kasperLogo.png")


def render_kasper_header() -> None:
    logo_path = page_icon_path()
    st.markdown(
        """
        <style>
        .kasper-header {
            display: flex;
            align-items: center;
            gap: 14px;
            margin: 0 0 18px 0;
        }
        .kasper-header img {
            width: 58px;
            height: 58px;
            object-fit: contain;
        }
        .kasper-header h1 {
            margin: 0;
            line-height: 1;
            font-size: 3.1rem;
            letter-spacing: 0;
            font-weight: 800;
        }
        @media (max-width: 640px) {
            .kasper-header img {
                width: 44px;
                height: 44px;
            }
            .kasper-header h1 {
                font-size: 2.35rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="kasper-header">
            <img src="{_image_data_uri(logo_path)}" alt="Kasper logo" />
            <h1>Kasper</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


@lru_cache(maxsize=4)
def _image_data_uri(path: str) -> str:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix.lstrip('.') or 'png'}"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@lru_cache(maxsize=1)
def _page_icon_data_uri() -> str:
    return _image_data_uri(page_icon_path())


def apply_branding_head() -> None:
    icon_href = _page_icon_data_uri()
    components.html(
        f"""
        <script>
        const iconHref = "{icon_href}";
        const head = window.parent.document.head;
        const linkDefs = [
          {{ rel: "icon", type: "image/png", sizes: "32x32" }},
          {{ rel: "shortcut icon", type: "image/png" }},
          {{ rel: "apple-touch-icon", type: "image/png", sizes: "180x180" }}
        ];

        for (const attrs of linkDefs) {{
          const selector = `link[rel="${{attrs.rel}}"]`;
          let link = head.querySelector(selector);
          if (!link) {{
            link = window.parent.document.createElement("link");
            head.appendChild(link);
          }}
          link.rel = attrs.rel;
          link.type = attrs.type;
          if (attrs.sizes) {{
            link.sizes = attrs.sizes;
          }}
          link.href = iconHref;
        }}
        </script>
        """,
        height=0,
        width=0,
    )


def apply_sidebar_nav_styles() -> None:
    logo_uri = _page_icon_data_uri()
    st.markdown(
        f"""
        <style>
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #f8fafc 0%, #fffdf8 52%, #f6f8fb 100%);
            border-right: 1px solid rgba(16, 37, 66, 0.08);
        }}
        section[data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.15rem;
        }}
        section[data-testid="stSidebar"] nav {{
            padding-top: 4.65rem;
            position: relative;
        }}
        section[data-testid="stSidebar"] nav::before {{
            content: "";
            position: absolute;
            top: 0.3rem;
            left: 0.9rem;
            width: 2.55rem;
            height: 2.55rem;
            border-radius: 8px;
            background: url("{logo_uri}") center / contain no-repeat;
            box-shadow: 0 8px 20px rgba(16, 37, 66, 0.08);
        }}
        section[data-testid="stSidebar"] nav::after {{
            content: "Kasper";
            position: absolute;
            top: 0.72rem;
            left: 4.05rem;
            color: #102542;
            font-size: 1.08rem;
            font-weight: 800;
            letter-spacing: 0;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] {{
            gap: 0.15rem;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] {{
            display: none;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNavSectionHeader"] {{
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0.85rem 0 0.35rem 0.15rem;
            padding: 0.15rem 0.35rem;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNavSection"] + [data-testid="stSidebarNavSection"] {{
            margin-top: 0.8rem;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(16, 37, 66, 0.10);
        }}
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {{
            border-radius: 8px;
            margin: 0.08rem 0.25rem;
            padding: 0.48rem 0.68rem;
            color: #1f2937;
            transition: background-color 140ms ease, color 140ms ease, box-shadow 140ms ease;
        }}
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {{
            background: rgba(16, 37, 66, 0.07);
            color: #102542;
        }}
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"],
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-selected="true"],
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"].active {{
            background: #102542;
            color: #ffffff;
            box-shadow: 0 8px 18px rgba(16, 37, 66, 0.16);
        }}
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] span,
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-selected="true"] span,
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"].active span {{
            color: #ffffff;
            font-weight: 750;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNavSection"]:nth-of-type(2) a[data-testid="stSidebarNavLink"] {{
            color: #334155;
            font-size: 0.96rem;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNavSection"]:nth-of-type(2) a[data-testid="stSidebarNavLink"]::before {{
            content: "";
            width: 0.42rem;
            height: 0.42rem;
            margin-right: 0.5rem;
            border-radius: 999px;
            background: rgba(151, 119, 255, 0.70);
            display: inline-block;
            vertical-align: 0.08rem;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNavSection"]:nth-of-type(2) a[data-testid="stSidebarNavLink"][aria-current="page"]::before,
        section[data-testid="stSidebar"] [data-testid="stSidebarNavSection"]:nth-of-type(2) a[data-testid="stSidebarNavLink"][aria-selected="true"]::before {{
            background: #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
